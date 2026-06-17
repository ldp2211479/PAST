#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except AttributeError as exc:
        if "extra_special_tokens" not in str(exc) and "object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            extra_special_tokens={},
        )


def load_entity_map(dataset_name: str, base_data_dir: str) -> Dict[str, int]:
    path = os.path.join(base_data_dir, dataset_name, "entity2id.txt")
    entity_map: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name, entity_id = parts[:2]
            entity_map[name.replace("_", " ")] = int(entity_id)
    return entity_map


def build_entity_trie(entity_map: Dict[str, int], tokenizer) -> Dict:
    trie_root: Dict = {}
    for entity_name, entity_id in tqdm(entity_map.items(), desc="build trie"):
        token_ids = tokenizer.encode(entity_name, add_special_tokens=False)
        if not token_ids:
            continue

        node = trie_root
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node["__END__"] = entity_id
    return trie_root


def load_code_token_map(path: str) -> Dict[Tuple[int, int], str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    token_by_level_code: Dict[Tuple[int, int], str] = {}
    for token, meta in payload.items():
        level = int(meta["level"])
        code_index = int(meta["code_index"])
        token_by_level_code[(level, code_index)] = token
    return token_by_level_code


def _z_column_sort_key(column: str) -> int:
    if not column.startswith("z"):
        raise ValueError(f"Invalid semantic-code column name: {column}")
    return int(column[1:])


def load_entity_semantic_codes(
    entity_codes_path: str,
    code_token_map_path: str,
) -> Dict[int, Tuple[str, ...]]:
    token_by_level_code = load_code_token_map(code_token_map_path)
    out: Dict[int, Tuple[str, ...]] = {}

    with open(entity_codes_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in entity code file: {entity_codes_path}")
        z_columns = sorted(
            [name for name in reader.fieldnames if name.startswith("z")],
            key=_z_column_sort_key,
        )
        if not z_columns:
            raise ValueError(f"No z* semantic-code columns found in: {entity_codes_path}")

        for row in reader:
            entity_id = int(row["entity_id"])
            tokens: List[str] = []
            for level, column in enumerate(z_columns, start=1):
                code_index = int(row[column])
                token = token_by_level_code.get((level, code_index))
                if token is None:
                    raise ValueError(
                        f"No semantic-code token for entity_id={entity_id}, "
                        f"level={level}, code_index={code_index}."
                    )
                tokens.append(token)
            out[entity_id] = tuple(tokens)

    if not out:
        raise ValueError(f"No entity semantic codes loaded from: {entity_codes_path}")
    return out


def semantic_code_tokens_to_ids(entity_codes: Dict[int, Tuple[str, ...]], tokenizer) -> Dict[int, Tuple[int, ...]]:
    entity_code_ids: Dict[int, Tuple[int, ...]] = {}
    for entity_id, code_tokens in entity_codes.items():
        token_ids: List[int] = []
        for token in code_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or int(token_id) < 0:
                raise ValueError(
                    f"Tokenizer does not contain semantic-code token {token!r}. "
                    "Use the same tokenizer/model that was warmed with these code tokens."
                )
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1 or int(encoded[0]) != int(token_id):
                raise ValueError(
                    f"Semantic-code token must be atomic in tokenizer: {token}, encoded={encoded}"
                )
            token_ids.append(int(token_id))
        entity_code_ids[entity_id] = tuple(token_ids)
    return entity_code_ids


def build_semantic_code_trie(entity_code_ids: Dict[int, Tuple[int, ...]]) -> Dict[Any, Any]:
    trie_root: Dict[Any, Any] = {}
    for entity_id, token_ids in tqdm(entity_code_ids.items(), desc="build semantic-code trie"):
        node = trie_root
        for token_id in token_ids:
            node = node.setdefault(int(token_id), {})
        node.setdefault("__END__", []).append(int(entity_id))
    return trie_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build entity-name or semantic-code trie for DGS-Debias.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["entity_name", "semantic_code"],
        default="entity_name",
        help="entity_name builds a trie from entity2id.txt; semantic_code builds a trie from entity semantic codes.",
    )
    parser.add_argument("--dataset_name", type=str, default="ICEWS14")
    parser.add_argument("--base_data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--tokenizer_path", type=str, default="./models/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--semantic_code_dir",
        type=str,
        default=None,
        help="Directory containing entity_codes.tsv and code_token_map.json.",
    )
    parser.add_argument("--entity_codes_path", type=str, default=None)
    parser.add_argument("--code_token_map_path", type=str, default=None)
    parser.add_argument("--trie_filename", type=str, default="entity_trie.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)

    if args.mode == "entity_name":
        entity_map = load_entity_map(args.dataset_name, args.base_data_dir)
        entity_trie = build_entity_trie(entity_map, tokenizer)

        with open(os.path.join(save_dir, "clean_entity_map.json"), "w", encoding="utf-8") as f:
            json.dump(entity_map, f, ensure_ascii=False, indent=2)
    else:
        if args.semantic_code_dir:
            entity_codes_path = args.entity_codes_path or os.path.join(args.semantic_code_dir, "entity_codes.tsv")
            code_token_map_path = args.code_token_map_path or os.path.join(args.semantic_code_dir, "code_token_map.json")
        else:
            entity_codes_path = args.entity_codes_path
            code_token_map_path = args.code_token_map_path
        if not entity_codes_path or not code_token_map_path:
            raise ValueError(
                "semantic_code mode requires --semantic_code_dir, or both "
                "--entity_codes_path and --code_token_map_path."
            )

        entity_codes = load_entity_semantic_codes(entity_codes_path, code_token_map_path)
        entity_code_ids = semantic_code_tokens_to_ids(entity_codes, tokenizer)
        entity_trie = build_semantic_code_trie(entity_code_ids)

        metadata = {
            "mode": args.mode,
            "dataset_name": args.dataset_name,
            "entity_codes_path": entity_codes_path,
            "code_token_map_path": code_token_map_path,
            "tokenizer_path": args.tokenizer_path,
            "num_entities": len(entity_code_ids),
            "num_levels": len(next(iter(entity_code_ids.values()))),
        }
        with open(os.path.join(save_dir, "semantic_code_trie_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    trie_path = os.path.join(save_dir, args.trie_filename)
    with open(trie_path, "wb") as f:
        pickle.dump(entity_trie, f)
    print(f"Saved trie: {trie_path}")


if __name__ == "__main__":
    main()
