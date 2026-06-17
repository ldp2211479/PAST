import logging
import pickle
from typing import Callable, Dict, List, Set, Tuple

import torch

logger = logging.getLogger(__name__)

END_KEY = "__END__"


class TrieConstraintFactory:
    def __init__(self, trie_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        logger.info("Loading entity trie from %s", trie_path)
        with open(trie_path, "rb") as f:
            self.trie = pickle.load(f)

        self.global_prefix_counts: Dict[Tuple[int, ...], int] = {}
        self.global_terminal_prefixes: Set[Tuple[int, ...]] = set()
        self.global_next_token_counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self._build_global_stats(self.trie, ())

    def _build_global_stats(self, node: Dict, prefix: Tuple[int, ...]) -> int:
        total = 1 if END_KEY in node else 0
        next_token_counts: Dict[int, int] = {}
        for token_id, child in node.items():
            if token_id == END_KEY:
                continue
            child_prefix = prefix + (int(token_id),)
            child_total = self._build_global_stats(child, child_prefix)
            next_token_counts[int(token_id)] = child_total
            total += child_total

        self.global_prefix_counts[prefix] = total
        self.global_next_token_counts[prefix] = next_token_counts
        if END_KEY in node:
            self.global_terminal_prefixes.add(prefix)
        return total

    def _get_global_allowed_tokens(self, entity_prefix: Tuple[int, ...]) -> List[int]:
        if entity_prefix not in self.global_prefix_counts:
            return [self.eos_token_id] if self.eos_token_id is not None else []

        allowed_tokens = list(self.global_next_token_counts.get(entity_prefix, {}).keys())
        if entity_prefix in self.global_terminal_prefixes and self.eos_token_id is not None:
            allowed_tokens.append(self.eos_token_id)
        if not allowed_tokens and self.eos_token_id is not None:
            return [self.eos_token_id]
        return allowed_tokens

    def get_constraint_fn(self, prompt_len: int) -> Callable[[int, torch.Tensor], List[int]]:
        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            if input_ids.dim() == 1:
                generated_ids = input_ids[prompt_len:].tolist()
            else:
                generated_ids = input_ids[batch_id, prompt_len:].tolist()

            return self._get_global_allowed_tokens(tuple(generated_ids))

        return prefix_allowed_tokens_fn
