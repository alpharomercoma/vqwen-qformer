"""VQwenQFormerForCausalLM — MiniGPT-4-style adapter on top of Qwen3-4B.

  Image (224x224)
    -> EVA-ViT-G/14         [FROZEN, from BLIP-2-opt-2.7b]  -> (B, 257, 1408)
    -> Q-Former (12 layers) [FROZEN, from BLIP-2-opt-2.7b]  -> (B, 32, 768)
       (queries pretrained on 129M image-text pairs + co-trained with OPT-2.7B)
    -> Linear(768, 2560)    [TRAINABLE, 2M params]           -> (B, 32, 2560)
  -> splice into Qwen3-4B input embeddings at <image> sentinel
  -> Qwen3-4B LM            [FROZEN stage-1; LoRA-adapted stage-2]

Loads the frozen vision + Q-Former + query_tokens from a compact extracted
bundle (see scripts/01_download_blip2.py). The bundle contains Blip2 sub-module
state_dicts only; we instantiate matching sub-modules from Blip2Config and
load state_dicts strictly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Blip2Config,
    Blip2QFormerModel,
    Blip2VisionModel,
    Qwen3ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, QFORMER_HIDDEN_SIZE
from .projector import LinearProjector


class VQwenQFormerForCausalLM(nn.Module):
    def __init__(
        self,
        blip2_bundle_path: str | Path,       # models/blip2-frozen/
        llm_model_path: str | Path,          # models/Qwen3-4B/
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
        use_liger: bool = False,
        skip_vision_stack: bool = False,     # True when training on cached Q-Former features
    ):
        super().__init__()
        bundle = Path(blip2_bundle_path)

        blip2_cfg = Blip2Config.from_pretrained(str(bundle))
        num_q = blip2_cfg.num_query_tokens
        q_hidden = blip2_cfg.qformer_config.hidden_size

        if skip_vision_stack:
            # Cached-features mode: skip loading ViT-G + Q-Former entirely.
            # Saves ~1.2 GB VRAM + ~2 s model-load time.
            self.vision_model = None
            self.qformer = None
            self.query_tokens = None
        else:
            self.vision_model = Blip2VisionModel(blip2_cfg.vision_config).to(dtype=dtype)
            self.qformer = Blip2QFormerModel(blip2_cfg.qformer_config).to(dtype=dtype)
            self.query_tokens = nn.Parameter(torch.zeros(1, num_q, q_hidden, dtype=dtype))
            v_state = torch.load(bundle / "vision_model.bin", map_location="cpu", weights_only=True)
            self.vision_model.load_state_dict(v_state, strict=True)
            q_state = torch.load(bundle / "qformer.bin", map_location="cpu", weights_only=True)
            self.qformer.load_state_dict(q_state, strict=True)
            qt = torch.load(bundle / "query_tokens.bin", map_location="cpu", weights_only=True)
            self.query_tokens.data.copy_(qt["query_tokens"].to(dtype=dtype))
            for p in self.vision_model.parameters(): p.requires_grad_(False)
            for p in self.qformer.parameters(): p.requires_grad_(False)
            self.query_tokens.requires_grad_(False)
            self.vision_model.eval(); self.qformer.eval()

        # --- Qwen3-4B (frozen stage-1; LoRA-adapted stage-2) ---
        if use_liger:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=False, fused_linear_cross_entropy=True,
            )
        self.llm = Qwen3ForCausalLM.from_pretrained(
            str(llm_model_path), dtype=dtype, attn_implementation=attn_implementation,
        )
        for p in self.llm.parameters(): p.requires_grad_(False)

        # --- Linear projector (the only trainable module stage-1) ---
        llm_hidden = self.llm.config.hidden_size              # 2560 for Qwen3-4B
        assert q_hidden == QFORMER_HIDDEN_SIZE, f"expected q_hidden={QFORMER_HIDDEN_SIZE}, got {q_hidden}"
        self.projector = LinearProjector(q_hidden, llm_hidden).to(dtype=dtype)
        self.num_query_tokens = num_q
        self._skip_vision_stack = skip_vision_stack

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_model is not None:
            self.vision_model.eval()
        if self.qformer is not None:
            self.qformer.eval()
        return self

    def enable_llm_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 257, 1408)."""
        with torch.no_grad():
            vdtype = next(self.vision_model.parameters()).dtype
            out = self.vision_model(pixel_values.to(dtype=vdtype))
        return out.last_hidden_state

    def qformer_summarize(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """(B, 257, 1408) -> (B, 32, 768)."""
        B = image_embeds.size(0)
        queries = self.query_tokens.expand(B, -1, -1).to(dtype=image_embeds.dtype)
        attn_mask = torch.ones(image_embeds.shape[:2], dtype=torch.long, device=image_embeds.device)
        with torch.no_grad():
            out = self.qformer(
                query_embeds=queries,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=attn_mask,
            )
        return out.last_hidden_state[:, : self.num_query_tokens, :]

    def _project_and_splice(self, input_ids, attention_mask, labels,
                            pixel_values=None, qformer_features=None):
        """Fast-path if `qformer_features` provided (pre-cached offline): skips
        the frozen vision+qformer forward and runs only the trainable projector."""
        if qformer_features is not None:
            q = qformer_features.to(dtype=self.projector.fc.weight.dtype)
        else:
            img = self.encode_images(pixel_values)          # (B, 257, 1408)
            q = self.qformer_summarize(img)                 # (B, 32, 768)
        proj = self.projector(q)                            # (B, 32, 2560)
        embed_tokens = self.llm.get_input_embeddings()
        B = input_ids.size(0); device = input_ids.device
        Nq = proj.size(1)

        new_embeds, new_labels, new_lens = [], [], []
        for b in range(B):
            real_len = int(attention_mask[b].sum().item())
            ids = input_ids[b, :real_len]
            lab = labels[b, :real_len]
            img_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if img_positions.numel() == 0:
                emb = embed_tokens(ids.clamp(min=0))
                new_embeds.append(emb); new_labels.append(lab); new_lens.append(emb.size(0)); continue
            assert img_positions.numel() == 1
            p = int(img_positions[0].item())
            pre_ids, post_ids = ids[:p], ids[p + 1:]
            pre_lab, post_lab = lab[:p], lab[p + 1:]
            pre_emb = embed_tokens(pre_ids.clamp(min=0)) if pre_ids.numel() else \
                torch.empty(0, proj.size(-1), dtype=proj.dtype, device=device)
            post_emb = embed_tokens(post_ids.clamp(min=0)) if post_ids.numel() else \
                torch.empty(0, proj.size(-1), dtype=proj.dtype, device=device)
            img_emb = proj[b].to(dtype=pre_emb.dtype if pre_emb.numel() else post_emb.dtype)
            img_lab = torch.full((Nq,), IGNORE_INDEX, dtype=lab.dtype, device=device)
            new_embeds.append(torch.cat([pre_emb, img_emb, post_emb], dim=0))
            new_labels.append(torch.cat([pre_lab, img_lab, post_lab], dim=0))
            new_lens.append(new_embeds[-1].size(0))

        max_len = max(new_lens); hidden = new_embeds[0].size(1); dtype = new_embeds[0].dtype
        padded_emb = torch.zeros(B, max_len, hidden, dtype=dtype, device=device)
        padded_lab = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        padded_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, (e, l, L) in enumerate(zip(new_embeds, new_labels, new_lens)):
            padded_emb[b, :L] = e; padded_lab[b, :L] = l; padded_mask[b, :L] = 1
        return padded_emb, padded_mask, padded_lab

    def forward(self, input_ids, attention_mask, labels,
                pixel_values=None, qformer_features=None, **kwargs):
        inputs_embeds, new_mask, new_labels = self._project_and_splice(
            input_ids, attention_mask, labels,
            pixel_values=pixel_values, qformer_features=qformer_features,
        )
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=new_mask,
                        labels=new_labels, return_dict=True)


def build_tokenizer(llm_path):
    tok = AutoTokenizer.from_pretrained(str(llm_path), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_image_processor(blip2_bundle_path):
    """Load the BlipImageProcessor that was saved alongside the compact bundle (224x224)."""
    return AutoImageProcessor.from_pretrained(str(blip2_bundle_path))


def apply_lora_to_llm(model, r: int = 128, lora_alpha: int = 256, lora_dropout: float = 0.05,
                      target_modules: Optional[list] = None):
    from peft import LoraConfig, get_peft_model
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                     target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
    model.llm = get_peft_model(model.llm, cfg)
    return model


def load_stage1_projector(model, projector_path):
    state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
    state = {k.removeprefix("projector."): v for k, v in state.items()}
    model.projector.load_state_dict(state, strict=True)
