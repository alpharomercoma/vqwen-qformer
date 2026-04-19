"""HF Trainer wrapper. Stage-1 trains only Linear; stage-2 adds LoRA on Qwen3."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

from .dataset import (
    DataCollatorForSupervisedDataset,
    LlavaInstructCachedDataset,
    LlavaInstructDataset,
    LlavaPretrainDataset,
)
from .model import (
    VQwenQFormerForCausalLM,
    apply_lora_to_llm,
    build_image_processor,
    build_tokenizer,
    load_stage1_projector,
)


class JsonlLossLogger(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        if "loss" in logs:
            self.f.write(json.dumps({
                "step": state.global_step, "epoch": state.epoch,
                "wall_s": round(time.time() - self.t0, 2),
                "loss": logs.get("loss"), "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
            }) + "\n")
        if "eval_loss" in logs:
            self.f.write(json.dumps({
                "step": state.global_step, "epoch": state.epoch,
                "wall_s": round(time.time() - self.t0, 2),
                "eval_loss": logs.get("eval_loss"),
            }) + "\n")


class CopyBestCheckpointCallback(TrainerCallback):
    """At end of training, copy files from `state.best_model_checkpoint` over
    the top-level output_dir. Works around HF's `load_best_model_at_end` which
    requires a standard pytorch_model.bin that our `_save` does not produce.

    Our checkpoint layout is: projector.bin + lora_adapter/ + config.yaml.
    """
    def on_train_end(self, args, state, control, **kwargs):
        import shutil
        if not state.best_model_checkpoint:
            print("[best] no best_model_checkpoint recorded; keeping last.")
            return
        src = Path(state.best_model_checkpoint)
        dst = Path(args.output_dir)
        if not src.exists():
            print(f"[best] src {src} missing; keeping last.")
            return
        for fname in ("projector.bin", "config.yaml"):
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
        lora_src = src / "lora_adapter"
        if lora_src.exists():
            if (dst / "lora_adapter").exists():
                shutil.rmtree(dst / "lora_adapter")
            shutil.copytree(lora_src, dst / "lora_adapter")
        print(f"[best] promoted {src.name} (metric={state.best_metric:.4f}) -> {dst}")


class QFormerTrainer(Trainer):
    def __init__(self, *args, train_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_cfg = train_cfg or {}
        self.model_accepts_loss_kwargs = False

    def _load_best_model(self):
        # No-op: our custom `_save` uses a non-HF layout (projector.bin +
        # lora_adapter/). The HF default `_load_best_model` looks for
        # pytorch_model.bin and warns+skips if absent. We instead copy the
        # best checkpoint files at train_end via CopyBestCheckpointCallback.
        return

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if getattr(self.args, "group_by_length", False) and hasattr(ds, "lengths"):
            bs = self.args.train_batch_size * max(1, self.args.gradient_accumulation_steps)
            return LengthGroupedSampler(bs, dataset=ds, lengths=list(ds.lengths))
        return super()._get_train_sampler(train_dataset)

    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        lr = self.args.learning_rate
        projector_lr = self.train_cfg.get("projector_lr", lr)
        wd = self.args.weight_decay
        no_decay = ("bias", "LayerNorm.weight", "norm.weight")

        groups = {
            "projector_decay":  {"params": [], "lr": projector_lr, "weight_decay": wd},
            "projector_nodecay":{"params": [], "lr": projector_lr, "weight_decay": 0.0},
            "other_decay":      {"params": [], "lr": lr,           "weight_decay": wd},
            "other_nodecay":    {"params": [], "lr": lr,           "weight_decay": 0.0},
        }
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            bucket = "projector" if n.startswith("projector.") else "other"
            key = bucket + ("_nodecay" if any(nd in n for nd in no_decay) else "_decay")
            groups[key]["params"].append(p)
        param_groups = [g for g in groups.values() if g["params"]]
        self.optimizer = torch.optim.AdamW(
            param_groups, betas=(0.9, 0.999), eps=1e-8,
            fused=torch.cuda.is_available(),
        )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model
        pj_state = {f"projector.{k}": v for k, v in model.projector.state_dict().items()}
        torch.save(pj_state, os.path.join(output_dir, "projector.bin"))
        if hasattr(model.llm, "peft_config"):
            model.llm.save_pretrained(os.path.join(output_dir, "lora_adapter"))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(self.train_cfg, f)


def _build_dataset(cfg, tokenizer, image_processor, num_query_tokens):
    fmt = cfg.get("conversation_format", "plain")
    cached_dir = cfg.get("cached_features_dir")
    if cached_dir:
        assert fmt == "qwen_chat", "cached features only implemented for qwen_chat stage-2"
        return LlavaInstructCachedDataset(
            cfg["json_path"], cached_dir, tokenizer,
            max_length=cfg.get("model_max_length", 2048),
            num_query_tokens=num_query_tokens,
        )
    if fmt == "plain":
        return LlavaPretrainDataset(cfg["json_path"], cfg["image_root"], tokenizer,
                                     image_processor, max_length=cfg.get("model_max_length", 2048))
    return LlavaInstructDataset(cfg["json_path"], cfg["image_root"], tokenizer,
                                 image_processor, max_length=cfg.get("model_max_length", 2048),
                                 num_query_tokens=num_query_tokens)


def build_model_and_processors(cfg):
    tokenizer = build_tokenizer(cfg["llm_model_path"])
    image_processor = build_image_processor(cfg["blip2_bundle_path"])
    model = VQwenQFormerForCausalLM(
        blip2_bundle_path=cfg["blip2_bundle_path"],
        llm_model_path=cfg["llm_model_path"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        use_liger=cfg.get("use_liger", False),
        skip_vision_stack=bool(cfg.get("cached_features_dir")),
    )
    return model, tokenizer, image_processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--loss_log_jsonl", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--save_strategy", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.logging_steps is not None: cfg["logging_steps"] = args.logging_steps
    if args.save_strategy is not None: cfg["save_strategy"] = args.save_strategy

    model, tokenizer, image_processor = build_model_and_processors(cfg)

    if cfg.get("load_stage1_projector"):
        print(f"[train] loading stage-1 projector from {cfg['load_stage1_projector']}")
        load_stage1_projector(model, cfg["load_stage1_projector"])

    if cfg.get("gradient_checkpointing", False):
        model.enable_llm_gradient_checkpointing()

    if cfg.get("use_lora", False):
        model = apply_lora_to_llm(
            model,
            r=cfg.get("lora_r", 128),
            lora_alpha=cfg.get("lora_alpha", 256),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules"),
        )

    # Projector is always trainable. Vision + Q-Former + queries remain frozen.
    for p_ in model.projector.parameters(): p_.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable: {n_train/1e6:.2f}M / {n_all/1e6:.2f}M total")

    train_ds = _build_dataset(cfg, tokenizer, image_processor,
                              num_query_tokens=model.num_query_tokens)
    eval_ds = None
    if cfg.get("eval_json_path"):
        eval_cfg = dict(cfg); eval_cfg["json_path"] = cfg["eval_json_path"]
        eval_ds = _build_dataset(eval_cfg, tokenizer, image_processor,
                                  num_query_tokens=model.num_query_tokens)
        print(f"[train] eval dataset: {len(eval_ds)} samples from {cfg['eval_json_path']}")
    collator = DataCollatorForSupervisedDataset(pad_token_id=tokenizer.pad_token_id)

    args_hf = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 64),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=False,
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 5),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 12),
        dataloader_persistent_workers=cfg.get("dataloader_persistent_workers", True),
        dataloader_pin_memory=cfg.get("dataloader_pin_memory", True),
        dataloader_prefetch_factor=cfg.get("dataloader_prefetch_factor", 4),
        optim=cfg.get("optim", "adamw_torch_fused"),
        report_to=cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        group_by_length=cfg.get("group_by_length", False),
        eval_strategy=cfg.get("eval_strategy", "no") if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps"),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size",
                                           cfg.get("per_device_train_batch_size", 32)),
        load_best_model_at_end=bool(eval_ds) and cfg.get("load_best_model_at_end", False),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
    )

    callbacks = []
    if args.loss_log_jsonl:
        callbacks.append(JsonlLossLogger(args.loss_log_jsonl))
    if eval_ds is not None and cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(cfg["early_stopping_patience"]),
        ))
    if eval_ds is not None and cfg.get("load_best_model_at_end", False):
        callbacks.append(CopyBestCheckpointCallback())

    trainer = QFormerTrainer(
        model=model, args=args_hf, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, callbacks=callbacks, train_cfg=cfg,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model()
    print("[train] done.")


if __name__ == "__main__":
    main()
