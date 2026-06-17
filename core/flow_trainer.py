import json
import logging
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import TrainerCallback, set_seed

from wise.core.utils_trie import END_KEY


logger = logging.getLogger(__name__)


class SFTJsonDataset(Dataset):
    def __init__(self, items: List[Dict[str, List[int]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.items[index]


@dataclass
class SFTDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        pad_token_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }



@dataclass
class MSLDataCollator:
    tokenizer: Any
    entity_trie_path: str
    pad_to_multiple_of: Optional[int] = 8

    def __post_init__(self) -> None:
        with open(self.entity_trie_path, "rb") as f:
            self.entity_trie = pickle.load(f)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self._allowed_tokens_cache: Dict[tuple, List[int]] = {}

    def _get_allowed_tokens(self, prefix_tokens: List[int]) -> List[int]:
        prefix_key = tuple(int(x) for x in prefix_tokens)
        cached = self._allowed_tokens_cache.get(prefix_key)
        if cached is not None:
            return cached

        node = self.entity_trie
        for token_id in prefix_key:
            node = node.get(int(token_id))
            if node is None:
                allowed = [self.eos_token_id] if self.eos_token_id is not None else []
                self._allowed_tokens_cache[prefix_key] = allowed
                return allowed

        allowed = [int(token_id) for token_id in node.keys() if token_id != END_KEY]
        if END_KEY in node and self.eos_token_id is not None:
            allowed.append(int(self.eos_token_id))

        if not allowed and self.eos_token_id is not None:
            allowed = [int(self.eos_token_id)]

        self._allowed_tokens_cache[prefix_key] = allowed
        return allowed

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        flat_allowed_token_ids: List[List[int]] = []
        max_valid_tokens = 1

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)

            target_token_ids = [int(token_id) for token_id in feature["labels"] if int(token_id) != -100]
            prefix_token_ids: List[int] = []
            for token_id in target_token_ids:
                allowed_token_ids = self._get_allowed_tokens(prefix_token_ids)
                if token_id not in allowed_token_ids:
                    allowed_token_ids = allowed_token_ids + [token_id]
                flat_allowed_token_ids.append(allowed_token_ids)
                if len(allowed_token_ids) > max_valid_tokens:
                    max_valid_tokens = len(allowed_token_ids)
                prefix_token_ids.append(token_id)

        if flat_allowed_token_ids:
            allowed_token_ids = torch.full(
                (len(flat_allowed_token_ids), max_valid_tokens),
                fill_value=0,
                dtype=torch.long,
            )
            allowed_token_mask = torch.zeros(
                (len(flat_allowed_token_ids), max_valid_tokens),
                dtype=torch.bool,
            )
            for row_index, token_ids in enumerate(flat_allowed_token_ids):
                token_tensor = torch.tensor(token_ids, dtype=torch.long)
                valid_len = token_tensor.numel()
                allowed_token_ids[row_index, :valid_len] = token_tensor
                allowed_token_mask[row_index, :valid_len] = True
        else:
            allowed_token_ids = torch.zeros((0, 1), dtype=torch.long)
            allowed_token_mask = torch.zeros((0, 1), dtype=torch.bool)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "allowed_token_ids": allowed_token_ids,
            "allowed_token_mask": allowed_token_mask,
        }


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        self.tokenizer.save_pretrained(args.output_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.tokenizer.save_pretrained(args.output_dir)
        return control


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _length_bin_name(bin_index: int, bin_size: int) -> str:
    start = bin_index * bin_size + 1
    end = (bin_index + 1) * bin_size
    return f"{start}-{end}"


def encode_sft_dataset(
    json_path: str,
    tokenizer: Any,
    max_seq_len: int,
    max_samples: int = -1,
    length_bin_size: int = 256,
) -> SFTJsonDataset:
    raw_items = load_json(json_path)
    if max_samples is not None and max_samples > 0:
        # np.random.shuffle(raw_items)
        raw_items = raw_items[:max_samples]

    encoded_items: List[Dict[str, List[int]]] = []
    eos_token_id = tokenizer.eos_token_id
    length_counter: Counter[int] = Counter()
    filtered_over_max = 0

    for item in tqdm(raw_items):
        prompt_ids = tokenizer.encode(item["input"], add_special_tokens=False)
        output_ids = tokenizer.encode(item["output"], add_special_tokens=False)
        if eos_token_id is not None:
            output_ids = output_ids + [eos_token_id]

        total_len = len(prompt_ids) + len(output_ids)
        bin_index = max((total_len - 1) // max(length_bin_size, 1), 0)
        length_counter[bin_index] += 1

        if total_len > max_seq_len:
            filtered_over_max += 1
            continue

        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
        attention_mask = [1] * len(input_ids)

        encoded_items.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    logger.info(
        "Token length distribution for %s before filtering (bin_size=%d, max_seq_len=%d):",
        json_path,
        length_bin_size,
        max_seq_len,
    )
    for bin_index in sorted(length_counter):
        upper = (bin_index + 1) * max(length_bin_size, 1)
        mark = " [over max_seq_len bin]" if upper > max_seq_len else ""
        logger.info(
            "  %s: %d%s",
            _length_bin_name(bin_index, max(length_bin_size, 1)),
            length_counter[bin_index],
            mark,
        )
    logger.info(
        "Encoded %d / %d samples from %s; filtered_over_max_seq_len=%d",
        len(encoded_items),
        len(raw_items),
        json_path,
        filtered_over_max,
    )

    if len(encoded_items) == 0:
        raise ValueError(
            f"No samples left after filtering {json_path} with max_seq_len={max_seq_len}. "
            "Please increase --max_seq_len or reduce prompt/history length."
        )

    return SFTJsonDataset(encoded_items)
