"""Inference: load projector + (optional LoRA) + frozen Blip2 vision+Q-Former + Qwen3."""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image

from .constants import IGNORE_INDEX
from .model import VQwenQFormerForCausalLM, build_image_processor, build_tokenizer
from .preprocess import tokenizer_image_token


def load_trained_model(checkpoint_dir: str | Path,
                       attn_implementation: str = "sdpa",
                       dtype: torch.dtype = torch.bfloat16,
                       device: str = "cuda") -> VQwenQFormerForCausalLM:
    """Load a stage-1 or stage-2 checkpoint.

    Checkpoint layout (produced by QFormerTrainer._save):
      projector.bin                  # trainable Linear state_dict
      lora_adapter/                  # optional (stage-2 only)
      config.yaml                    # training yaml, contains blip2_bundle_path + llm_model_path
    """
    ckpt_dir = Path(checkpoint_dir)
    with open(ckpt_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = VQwenQFormerForCausalLM(
        blip2_bundle_path=cfg["blip2_bundle_path"],
        llm_model_path=cfg["llm_model_path"],
        attn_implementation=attn_implementation,
        dtype=dtype,
        use_liger=False,                # no need at inference
        skip_vision_stack=False,        # inference runs vision live
    )

    state = torch.load(ckpt_dir / "projector.bin", map_location="cpu", weights_only=True)
    state = {k.removeprefix("projector."): v for k, v in state.items()}
    model.projector.load_state_dict(state, strict=True)

    lora_dir = ckpt_dir / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(lora_dir))
        model.llm = model.llm.merge_and_unload()

    model.to(device=device).eval()
    return model


@torch.no_grad()
def generate_caption(model: VQwenQFormerForCausalLM,
                     tokenizer,
                     image_processor,
                     image_path: str | Path,
                     prompt: str = "<image>",
                     max_new_tokens: int = 64,
                     do_sample: bool = False,
                     chat_template: bool = False) -> str:
    device = next(model.llm.parameters()).device
    vision_dtype = next(model.vision_model.parameters()).dtype
    image = Image.open(image_path).convert("RGB")
    pv = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device=device, dtype=vision_dtype)

    if chat_template:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([tokenizer_image_token(prompt, tokenizer)], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    fake_labels = torch.full_like(input_ids, IGNORE_INDEX)

    inputs_embeds, attn_mask, _ = model._project_and_splice(
        input_ids, attn, fake_labels, pixel_values=pv
    )
    kw = dict(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
              max_new_tokens=max_new_tokens,
              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
              do_sample=do_sample)
    out = model.llm.generate(**kw)
    return tokenizer.decode(out[0], skip_special_tokens=True)
