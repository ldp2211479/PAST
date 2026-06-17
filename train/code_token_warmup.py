#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from wise.train.common import (
    CheckpointTokenizerCallback,
    TrainerLogCallback,
    add_code_tokens,
    build_training_args,
    freeze_except_embeddings_and_lm_head,
    get_dtype,
    load_code_tokens,
    sanitized_args,
    setup_logging,
)
from wise.core.flow_trainer import SFTDataCollator, encode_sft_dataset, set_global_seed


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up semantic code tokens before stage-1 SFT.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--semantic_code_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--target_mode", type=str, default="auto", choices=["auto", "code", "as_is"])
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--eval_strategy", type=str, default="no")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--safe_serialization", action="store_true")
    return parser.parse_args()


def normalize_entity_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").strip().split())


def load_level_code_to_token(semantic_code_dir: str) -> Dict[Tuple[int, int], str]:
    path = os.path.join(semantic_code_dir, "code_token_map.json")
    with open(path, "r", encoding="utf-8") as f:
        token_map = json.load(f)
    return {
        (int(meta["level"]), int(meta["code_index"])): token
        for token, meta in token_map.items()
    }


def load_entity_name_to_code(semantic_code_dir: str) -> Dict[str, str]:
    token_by_level_code = load_level_code_to_token(semantic_code_dir)
    path = os.path.join(semantic_code_dir, "entity_codes.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Entity semantic code file not found: {path}")

    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = normalize_entity_name(row["entity_name"])
            z_columns = sorted(
                (int(match.group(1)), column)
                for column in (reader.fieldnames or [])
                for match in [re.fullmatch(r"z(\d+)", column)]
                if match
            )
            if not z_columns:
                raise ValueError(f"No z0/z1/... code columns found in {path}.")
            codes = [int(row[column]) for _, column in z_columns]
            tokens = [
                token_by_level_code[(level, code)]
                for level, code in enumerate(codes, start=1)
            ]
            out[name] = "".join(tokens)
    return out


def looks_like_code(text: str) -> bool:
    text = str(text).strip()
    return text.startswith("<z") and text.endswith(">")


def build_code_target_json(args: argparse.Namespace) -> str:
    train_json = args.train_json or os.path.join(args.data_dir, "valid.json")
    with open(train_json, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    name_to_code = load_entity_name_to_code(args.semantic_code_dir)
    converted: List[Dict[str, Any]] = []
    missing: List[str] = []
    for item in raw_items:
        new_item = dict(item)
        output = str(item["output"]).strip()
        if args.target_mode == "as_is" or (args.target_mode == "auto" and looks_like_code(output)):
            target = output
        else:
            key = normalize_entity_name(output)
            target = name_to_code.get(key)
            if target is None:
                missing.append(output)
                continue
        new_item["output"] = target
        converted.append(new_item)

    if not converted:
        raise ValueError(f"No warmup samples could be built from {train_json}.")
    if missing:
        logger.warning("Skipped %d samples with missing semantic codes; first few: %s", len(missing), missing[:10])

    path = os.path.join(args.output_dir, "code_token_warmup_train.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    logger.info("Wrote code-token warmup data (%d samples)", len(converted))
    return path


def load_tokenizer(path: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True)
    except AttributeError as exc:
        if "extra_special_tokens" not in str(exc):
            raise
        logger.warning("Retrying tokenizer load with extra_special_tokens={} for older tokenizer_config compatibility.")
        return AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            extra_special_tokens={},
        )


def get_new_code_token_ids(tokenizer: Any, code_tokens: List[str], old_vocab: Dict[str, int]) -> List[int]:
    new_token_ids = []
    for token in code_tokens:
        if token in old_vocab:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            logger.warning("Could not resolve newly added code token to id: %s", token)
            continue
        new_token_ids.append(int(token_id))
    return sorted(set(new_token_ids))


def mask_trainable_token_rows(model: Any, token_ids: List[int]) -> Dict[str, Any]:
    """Allow gradients only for selected token rows in embeddings/lm_head."""
    token_ids = sorted(set(int(token_id) for token_id in token_ids))
    token_ids_cpu = torch.tensor(token_ids, dtype=torch.long)
    modules = [
        ("input_embeddings", model.get_input_embeddings()),
        ("output_embeddings", model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None),
    ]

    seen_param_ids = set()
    stats: Dict[str, Any] = {
        "trainable_token_rows": len(token_ids),
        "modules": {},
    }
    for name, module in modules:
        if module is None or not hasattr(module, "weight"):
            continue

        weight = module.weight
        if id(weight) in seen_param_ids:
            stats["modules"][name] = {"skipped": "tied_weight"}
            continue
        seen_param_ids.add(id(weight))

        vocab_size = int(weight.shape[0])
        valid_token_ids = token_ids_cpu[token_ids_cpu.ge(0) & token_ids_cpu.lt(vocab_size)]
        row_mask = torch.zeros((vocab_size, 1), dtype=torch.bool)
        if valid_token_ids.numel() > 0:
            row_mask[valid_token_ids] = True

        def hook(grad: torch.Tensor, mask: torch.Tensor = row_mask) -> torch.Tensor:
            return grad * mask.to(device=grad.device, dtype=grad.dtype)

        weight.register_hook(hook)
        stats["modules"][name] = {
            "vocab_size": vocab_size,
            "hidden_size": int(weight.shape[1]) if weight.ndim > 1 else 1,
            "trainable_rows": int(valid_token_ids.numel()),
            "masked_rows": int(vocab_size - valid_token_ids.numel()),
        }

    return stats


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_file = setup_logging(args.output_dir)
    logger.info("Starting semantic code-token warmup")
    logger.info("Logging to train.log")
    logger.info(json.dumps(sanitized_args(args), ensure_ascii=False, indent=2))

    dtype = get_dtype(args)
    tokenizer = load_tokenizer(args.base_model, args.trust_remote_code)
    old_vocab = dict(tokenizer.get_vocab())
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    code_tokens = load_code_tokens(args.semantic_code_dir)
    added = add_code_tokens(tokenizer, model, code_tokens)
    new_token_ids = get_new_code_token_ids(tokenizer, code_tokens, old_vocab)
    logger.info("Loaded %d semantic code tokens; newly added %d tokens.", len(code_tokens), added)
    logger.info("New semantic code token ids selected for training: %d", len(new_token_ids))
    if added != len(new_token_ids):
        logger.warning(
            "Tokenizer reported %d added tokens, but resolved %d trainable new-token ids.",
            added,
            len(new_token_ids),
        )

    trainable_stats = freeze_except_embeddings_and_lm_head(model)
    logger.info("Trainable parameter stats: %s", json.dumps(trainable_stats, sort_keys=True))
    token_row_stats = mask_trainable_token_rows(model, new_token_ids)
    logger.info("Token-row gradient mask stats: %s", json.dumps(token_row_stats, sort_keys=True))
    if args.weight_decay != 0.0:
        logger.warning(
            "Overriding weight_decay from %s to 0.0 so frozen token rows cannot move via decoupled weight decay.",
            args.weight_decay,
        )
        args.weight_decay = 0.0

    warmup_json = build_code_target_json(args)
    train_dataset = encode_sft_dataset(
        json_path=warmup_json,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_train_samples,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=SFTDataCollator(tokenizer=tokenizer),
        callbacks=[CheckpointTokenizerCallback(tokenizer), TrainerLogCallback()],
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    merged_dir = os.path.join(args.output_dir, "merged")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved code-token warmup model.")

    model.save_pretrained(merged_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Saved merged/full code-token warmup model.")


if __name__ == "__main__":
    main()
