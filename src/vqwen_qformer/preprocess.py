"""Same preprocessing as vqwen: tokenizer_image_token + plain/qwen_chat formats."""
from __future__ import annotations

import copy
from typing import Dict, List

import torch

from .constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


def tokenizer_image_token(prompt: str, tokenizer, image_token_index: int = IMAGE_TOKEN_INDEX) -> List[int]:
    chunks = [tokenizer(c, add_special_tokens=False).input_ids for c in prompt.split(DEFAULT_IMAGE_TOKEN)]
    out: List[int] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            out.append(image_token_index)
        out.extend(chunk)
    return out


def preprocess_plain(sources: List[List[Dict]], tokenizer, max_length: int = 2048) -> Dict[str, List[torch.Tensor]]:
    eos = tokenizer.eos_token or ""
    input_ids_list, labels_list = [], []
    for source in sources:
        assert len(source) == 2
        full_text = DEFAULT_IMAGE_TOKEN + source[1]["value"] + eos
        input_ids = tokenizer_image_token(full_text, tokenizer)
        human_ids = tokenizer_image_token(DEFAULT_IMAGE_TOKEN, tokenizer)
        labels = copy.deepcopy(input_ids)
        labels[: len(human_ids)] = [IGNORE_INDEX] * len(human_ids)
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    return {"input_ids": input_ids_list, "labels": labels_list}


def _tokenize_with_image_maybe(text: str, tokenizer, has_image: bool) -> List[int]:
    if has_image and DEFAULT_IMAGE_TOKEN in text:
        return tokenizer_image_token(text, tokenizer)
    return tokenizer(text, add_special_tokens=False).input_ids


def preprocess_qwen(sources: List[List[Dict]], tokenizer, has_image: bool, max_length: int = 2048) -> Dict[str, List[torch.Tensor]]:
    eos_newline = "<|im_end|>\n"
    suffix_ids = tokenizer(eos_newline, add_special_tokens=False).input_ids
    input_ids_list, labels_list = [], []
    for source in sources:
        if source and source[0]["from"] != "human":
            source = source[1:]
        input_ids: List[int] = []
        labels: List[int] = []
        for turn in source:
            role = "user" if turn["from"] == "human" else "assistant"
            prefix_ids = tokenizer(f"<|im_start|>{role}\n", add_special_tokens=False).input_ids
            content_ids = _tokenize_with_image_maybe(turn["value"], tokenizer, has_image)
            turn_ids = prefix_ids + content_ids + suffix_ids
            if role == "assistant":
                turn_labels = [IGNORE_INDEX] * len(prefix_ids) + list(content_ids) + list(suffix_ids)
            else:
                turn_labels = [IGNORE_INDEX] * len(turn_ids)
            turn_labels = [IGNORE_INDEX if tid == IMAGE_TOKEN_INDEX else lid for tid, lid in zip(turn_ids, turn_labels)]
            input_ids.extend(turn_ids)
            labels.extend(turn_labels)
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    return {"input_ids": input_ids_list, "labels": labels_list}
