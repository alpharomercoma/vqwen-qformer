"""Export the tiktok-LoRA checkpoint as a standard Blip2ForConditionalGeneration.

Users load with zero custom code and no trust_remote_code:

    from transformers import Blip2ForConditionalGeneration, AutoProcessor
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID,
                                                          dtype=torch.bfloat16,
                                                          device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

Approach:
 - Wrap our components (ViT-G + Q-Former + Linear + merged Qwen3-4B) in
   Blip2ForConditionalGeneration. Its forward (transformers >= 4.46) supports
   splice-in-place when `image_token_index` is set.
 - Use Blip2Processor with num_query_tokens=None so it does NOT auto-prepend
   image tokens (BLIP-native behavior). Instead we bake `<image> * 32` into
   the chat template at the image-content position, matching our `<image>\\n`
   training prompt layout.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqwen_qformer.model import build_tokenizer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiktok-lora")
    p.add_argument("--output_dir", default="export/vqwen-qformer-tiktok")
    p.add_argument("--blip2_bundle", default="models/blip2-frozen")
    p.add_argument("--image_token", default="<image>")
    p.add_argument("--dtype", default="bfloat16", choices=["float32","float16","bfloat16"])
    p.add_argument("--test_generate", action="store_true")
    p.add_argument("--test_image", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    ckpt = REPO_ROOT / args.checkpoint
    out = REPO_ROOT / args.output_dir
    bundle = REPO_ROOT / args.blip2_bundle
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if out.exists():
        if not args.force:
            sys.exit(f"[export] refusing to overwrite {out}; pass --force.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with open(ckpt / "config.yaml") as f:
        tcfg = yaml.safe_load(f)

    # Frozen components from the BLIP-2 bundle
    print("[export] loading frozen vision + qformer + queries ...", file=sys.stderr)
    from transformers import Blip2Config, Blip2QFormerModel, Blip2VisionModel
    blip2_cfg = Blip2Config.from_pretrained(str(bundle))
    vision_model = Blip2VisionModel(blip2_cfg.vision_config).to(dtype=dtype)
    qformer = Blip2QFormerModel(blip2_cfg.qformer_config).to(dtype=dtype)
    num_q = blip2_cfg.num_query_tokens
    vision_model.load_state_dict(
        torch.load(bundle / "vision_model.bin", map_location="cpu", weights_only=True), strict=True)
    qformer.load_state_dict(
        torch.load(bundle / "qformer.bin", map_location="cpu", weights_only=True), strict=True)
    query_tokens_tensor = torch.load(
        bundle / "query_tokens.bin", map_location="cpu", weights_only=True)["query_tokens"].to(dtype=dtype)

    # Qwen3 + merge LoRA
    print("[export] loading Qwen3-4B + merging LoRA ...", file=sys.stderr)
    from transformers import Qwen3ForCausalLM
    llm = Qwen3ForCausalLM.from_pretrained(tcfg["llm_model_path"], dtype=dtype, attn_implementation="sdpa")
    lora_dir = ckpt / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, str(lora_dir)).merge_and_unload()

    # Trained projector
    proj_state = torch.load(ckpt / "projector.bin", map_location="cpu", weights_only=True)
    proj_state = {k.removeprefix("projector."): v for k, v in proj_state.items()}

    # Tokenizer + <image>
    tokenizer = build_tokenizer(tcfg["llm_model_path"])
    tokenizer.add_special_tokens({"additional_special_tokens": [args.image_token]})
    image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    print(f"[export] image_token_id={image_token_id}; vocab now={len(tokenizer)}", file=sys.stderr)
    old_vocab = llm.get_input_embeddings().weight.size(0)
    if image_token_id >= old_vocab:
        llm.resize_token_embeddings(image_token_id + 1, mean_resizing=True)

    # Build Blip2Config. use_decoder_only_language_model makes the LM be
    # AutoModelForCausalLM (Qwen3).
    full_cfg = Blip2Config(
        vision_config=blip2_cfg.vision_config.to_dict(),
        qformer_config=blip2_cfg.qformer_config.to_dict(),
        text_config=llm.config.to_dict(),
        num_query_tokens=num_q,
        image_token_index=image_token_id,
    )
    full_cfg.use_decoder_only_language_model = True

    # Instantiate Blip2ForConditionalGeneration and populate
    print("[export] instantiating Blip2ForConditionalGeneration ...", file=sys.stderr)
    from transformers import Blip2ForConditionalGeneration
    model = Blip2ForConditionalGeneration(full_cfg).to(dtype=dtype)
    model.vision_model.load_state_dict(vision_model.state_dict(), strict=True)
    model.qformer.load_state_dict(qformer.state_dict(), strict=True)
    with torch.no_grad():
        model.query_tokens.data.copy_(query_tokens_tensor)
    model.language_projection.load_state_dict(
        {"weight": proj_state["fc.weight"], "bias": proj_state["fc.bias"]}, strict=True)
    missing, unexpected = model.language_model.load_state_dict(llm.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"lm load mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
    del vision_model, qformer, llm
    torch.cuda.empty_cache()

    # Chat template: we PRE-EXPAND <image> to 32 copies at the image-content
    # position so Blip2Processor does not need to auto-expand/prepend.
    # The `* num_q` is baked into the string here.
    image_block = (args.image_token * num_q) + "\n"
    qwen3_chat_template = (
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
        "{%- if message['content'] is string -%}"
        "{{- message['content'] -}}"
        "{%- else -%}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image' -%}"
        f"{image_block}"
        "{%- elif item['type'] == 'text' -%}"
        "{{- item['text'] -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{{- '<|im_end|>\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
    )
    tokenizer.chat_template = qwen3_chat_template

    from transformers import AutoImageProcessor, Blip2Processor
    image_processor = AutoImageProcessor.from_pretrained(str(bundle))
    # num_query_tokens=None disables the auto-prepend path inside Blip2Processor.
    processor = Blip2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_query_tokens=None,
    )
    processor.chat_template = qwen3_chat_template

    if args.test_generate:
        if not args.test_image: sys.exit("[export] --test_generate requires --test_image")
        img_path = Path(args.test_image)
        if not img_path.is_absolute(): img_path = REPO_ROOT / img_path
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        m = model.to("cuda").eval()
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Is this sludge content? Answer yes or no."}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        print(f"[smoke] prompt ({len(prompt)} chars, <image> count = {prompt.count(args.image_token)})", file=sys.stderr)
        inp = processor(text=prompt, images=img, return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        inp["pixel_values"] = inp["pixel_values"].to(dtype=dtype)
        n_img = int((inp["input_ids"] == image_token_id).sum().item())
        print(f"[smoke] input_ids image-token count = {n_img}", file=sys.stderr)
        with torch.no_grad():
            out_ids = m.generate(**inp, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        reply = processor.decode(out_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[smoke] reply: {reply!r}")
        model = m.cpu()
        torch.cuda.empty_cache()

    print(f"[export] saving -> {out}", file=sys.stderr)
    model.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    from transformers import GenerationConfig
    GenerationConfig(max_new_tokens=256, do_sample=False,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id).save_pretrained(out)
    print("[export] done.", file=sys.stderr)


if __name__ == "__main__":
    main()
