#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
import logging
import shutil
from typing import Optional, Any


logger = logging.getLogger(__name__)
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
PeftModel = None
PeftConfig = None


def load_merge_runtime() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, PeftModel, PeftConfig
    try:
        import torch as torch_module
        from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
        from transformers import AutoTokenizer as HFAutoTokenizer
        from peft import PeftModel as PeftModelClass
        from peft import PeftConfig as PeftConfigClass
    except ImportError as exc:
        raise ImportError(
            "Merging LoRA weights requires compatible torch, transformers, and peft installations."
        ) from exc

    torch = torch_module
    AutoModelForCausalLM = HFAutoModelForCausalLM
    AutoTokenizer = HFAutoTokenizer
    PeftModel = PeftModelClass
    PeftConfig = PeftConfigClass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")

    parser.add_argument("--base_model", type=str, required=True, help="Path to the full base/warmup model.")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA adapter model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged model.")

    parser.add_argument(
        "--tokenizer_source",
        type=str,
        default="base",
        choices=["auto", "base", "lora"],
        help=(
            "Where to load tokenizer from. "
            "'auto': prefer lora_model if tokenizer files exist, else base_model; "
            "'base': always use base_model; "
            "'lora': always use lora_model."
        ),
    )

    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe_serialization", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resize_vocab", action="store_true")

    return parser.parse_args()


def has_legacy_extra_special_tokens(path: str) -> bool:
    config_path = Path(path) / "tokenizer_config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(config.get("extra_special_tokens"), list)


def sanitize_tokenizer_config(path: str) -> bool:
    config_path = Path(path) / "tokenizer_config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(config.get("extra_special_tokens"), list):
        return False

    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    config.pop("extra_special_tokens", None)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.warning("Removed legacy list-valued extra_special_tokens from %s", config_path)
    return True


def load_tokenizer(path: str, trust_remote_code: bool, **kwargs: Any):
    sanitize_tokenizer_config(path)
    try:
        return AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except AttributeError:
        if not has_legacy_extra_special_tokens(path):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["extra_special_tokens"] = {}
        logger.warning(
            "Retrying tokenizer load with extra_special_tokens={} for tokenizer_config compatibility."
        )
        return AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            **retry_kwargs,
        )


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    file_path = os.path.join(output_dir, "merge.log")
    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    file_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


def get_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def has_tokenizer_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    names =[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    return any(os.path.exists(os.path.join(path, x)) for x in names)


def choose_tokenizer_path(args: argparse.Namespace) -> str:
    if args.tokenizer_source == "base":
        return args.base_model
    if args.tokenizer_source == "lora":
        return args.lora_model

    if has_tokenizer_files(args.lora_model):
        return args.lora_model
    return args.base_model


def get_vocab_size_from_model(model: Any) -> int:
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise ValueError("Failed to get input embeddings from model.")
    return int(emb.weight.shape[0])


def maybe_resize_embeddings(model: Any, tokenizer: Any, do_resize: bool = True) -> None:
    model_vocab_size = get_vocab_size_from_model(model)
    tokenizer_vocab_size = len(tokenizer)

    logger.info("Current model vocab size: %d", model_vocab_size)
    logger.info("Current tokenizer vocab size: %d", tokenizer_vocab_size)

    if model_vocab_size == tokenizer_vocab_size:
        logger.info("Model vocab size matches tokenizer vocab size. No resize needed.")
        return

    if not do_resize:
        raise ValueError(
            f"Vocab size mismatch: model={model_vocab_size}, tokenizer={tokenizer_vocab_size}, "
            "but --resize_vocab is disabled."
        )

    logger.warning(
        "Vocab size mismatch detected. Resizing token embeddings from %d to %d.",
        model_vocab_size,
        tokenizer_vocab_size,
    )
    # 此处可能导致模型维度由151936缩短截断至151667，但不影响正确合并，因为Adapter层也是151667
    if tokenizer_vocab_size > model_vocab_size:
        model.resize_token_embeddings(tokenizer_vocab_size)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Resize done. New model vocab size: %d", get_vocab_size_from_model(model))


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    load_merge_runtime()

    logger.info("Starting LoRA merge")
    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=2))

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"base_model not found: {args.base_model}")
    if not os.path.exists(args.lora_model):
        raise FileNotFoundError(f"lora_model not found: {args.lora_model}")

    dtype = get_dtype(args)
    tokenizer_path = choose_tokenizer_path(args)

    logger.info("Loading PEFT config from: %s", args.lora_model)
    peft_config = PeftConfig.from_pretrained(args.lora_model)
    logger.info("PEFT base_model_name_or_path in adapter_config: %s", peft_config.base_model_name_or_path)

    logger.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = load_tokenizer(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading base model from: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    maybe_resize_embeddings(model, tokenizer, do_resize=args.resize_vocab)

    logger.info("Loading LoRA adapter from: %s", args.lora_model)
    model = PeftModel.from_pretrained(
        model,
        args.lora_model,
    )

    logger.info("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    if hasattr(merged_model, "config") and tokenizer.pad_token_id is not None:
        merged_model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(merged_model.config, "tie_word_embeddings", False) and args.tie:
        logger.warning("Disabling tie_word_embeddings before saving the merged model.")
        merged_model.config.tie_word_embeddings = False
    
    logger.info("Merged model final vocab size verified: %d", get_vocab_size_from_model(merged_model))

    logger.info("Saving merged model to: %s", args.output_dir)
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    logger.info("Saving tokenizer to: %s", args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Merge completed successfully.")


if __name__ == "__main__":
    main()
