#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch

from wise.train.common import (
    CheckpointTokenizerCallback,
    TrainerLogCallback,
    build_training_args,
    get_dtype,
    load_model_with_unsloth,
    parse_args,
    sanitized_args,
    setup_logging,
)
from wise.core.flow_trainer import SFTDataCollator, encode_sft_dataset, set_global_seed

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not args.use_unsloth:
        raise ValueError("This script is configured to train with Unsloth only; keep --use_unsloth enabled.")

    train_json = args.train_json or os.path.join(args.data_dir, "train.json")
    valid_json = args.valid_json or os.path.join(args.data_dir, "valid.json")

    dtype = get_dtype(args)
    # Initialize Python logging only after Unsloth has finished loading the
    # model, because Unsloth may touch the root logger / handlers during import
    # or model construction and otherwise prevent train.log from receiving logs.
    model, tokenizer = load_model_with_unsloth(args, dtype)
    log_file = setup_logging(args.output_dir)

    logger.info("Starting WISE stage-1 CE/SFT training")
    logger.info("Logging to train.log")
    logger.info("Loaded base model with Unsloth.")
    logger.info(json.dumps(sanitized_args(args), ensure_ascii=False, indent=2))

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    logger.info("Loading train data.")
    train_dataset = encode_sft_dataset(
        json_path=train_json,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_train_samples,
    )

    eval_dataset = None
    if args.eval_strategy.lower() != "no" and os.path.exists(valid_json):
        logger.info("Loading valid data.")
        eval_dataset = encode_sft_dataset(
            json_path=valid_json,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_eval_samples,
        )
    else:
        args.eval_strategy = "no"

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer=tokenizer),
        callbacks=[CheckpointTokenizerCallback(tokenizer), TrainerLogCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    adapter_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved stage-1 CE/SFT LoRA adapter.")


if __name__ == "__main__":
    main()
