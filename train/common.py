#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from transformers import TrainerCallback, TrainingArguments

# Unsloth is required by the stage-1 training scripts. Keep logits available for
# compatibility with Unsloth's patched forward path and downstream logging.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WISE stage-1 LoRA model with standard CE/SFT loss.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--valid_json", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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

    parser.add_argument("--use_unsloth", action="store_true")
    parser.set_defaults(use_unsloth=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "train.log")
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(log_format))
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return file_path


def sanitized_args(args: argparse.Namespace) -> Dict[str, Any]:
    path_keys = {
        "base_model",
        "data_dir",
        "output_dir",
        "train_json",
        "valid_json",
        "resume_from_checkpoint",
        "semantic_code_dir",
    }
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in path_keys and value:
            out[key] = "<path>"
        else:
            out[key] = value
    return out


def get_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def add_special_tokens(tokenizer: Any, model: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if hasattr(model, "config") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_code_tokens(semantic_code_dir: str) -> List[str]:
    path = os.path.join(semantic_code_dir, "code_token_map.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Code token map file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        token_map = json.load(f)

    rows: List[Tuple[int, int, str]] = []
    for token, meta in token_map.items():
        rows.append((int(meta["level"]), int(meta["code_index"]), str(token)))
    rows.sort()
    return [token for _, _, token in rows]


def add_code_tokens(tokenizer: Any, model: Any, code_tokens: List[str]) -> int:
    before = len(tokenizer)
    added = tokenizer.add_tokens(code_tokens, special_tokens=False)
    if len(tokenizer) != before:
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "config"):
        model.config.vocab_size = len(tokenizer)
    add_special_tokens(tokenizer, model)
    return int(added)


def freeze_except_embeddings_and_lm_head(model: Any) -> Dict[str, int]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_param_ids = set()
    modules = [
        model.get_input_embeddings(),
        model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None,
    ]
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
            trainable_param_ids.add(id(param))

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    unique_trainable = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad and id(param) in trainable_param_ids
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "unique_trainable": int(unique_trainable),
    }


class CheckpointTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.tokenizer.save_pretrained(checkpoint_dir)
        return control


class TrainerLogCallback(TrainerCallback):
    """Mirror HuggingFace Trainer logs into train.log."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        if not getattr(state, "is_world_process_zero", True):
            return control
        serializable_logs = {}
        for key, value in logs.items():
            if isinstance(value, float):
                serializable_logs[key] = round(value, 8)
            else:
                serializable_logs[key] = value
        logging.getLogger("trainer").info(
            "Trainer log: %s",
            json.dumps(serializable_logs, ensure_ascii=False, sort_keys=True),
        )
        return control


def patch_unsloth_reorder_cache(model: Any) -> None:
    def _reorder_cache_patch(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    for obj in [model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)]:
        if obj is not None and not hasattr(obj, "_reorder_cache"):
            obj._reorder_cache = types.MethodType(_reorder_cache_patch, obj)


def load_model_with_unsloth(args: argparse.Namespace, dtype: Optional[torch.dtype]):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required by this training script. Please install unsloth in the training environment."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=dtype,
        load_in_4bit=False,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    add_special_tokens(tokenizer, model)

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.seed,
        modules_to_save=None,
    )
    patch_unsloth_reorder_cache(model)
    return model, tokenizer


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        average_tokens_across_devices=False,
    )
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = args.eval_strategy
    else:
        kwargs["evaluation_strategy"] = args.eval_strategy
    return TrainingArguments(**kwargs)
