#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

from wise.semantic.tkgr.rq_kmeans import (
    RQKMeans,
    build_code_to_entity_index,
    codes_to_strings,
    unique_code_rate,
)


def resolve_path(path_str: str, must_exist: bool = False) -> Path:
    path = Path(path_str).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def load_entity2id(path: Path) -> List[Tuple[int, str]]:
    records: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Invalid entity2id line in {path}: {line}")
            name, idx = parts[:2]
            records.append((int(idx), name))
    records.sort(key=lambda x: x[0])
    return records


def load_embeddings(path: Path, npz_key: str | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        embeddings = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        keys = payload.files
        if npz_key is None:
            if len(keys) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {keys}; pass --npz-key."
                )
            npz_key = keys[0]
        embeddings = payload[npz_key]
    else:
        raise ValueError("Only .npy and .npz embeddings are currently supported.")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}.")
    return embeddings.astype(np.float32, copy=False)


def align_embeddings_to_entities(
    embeddings: np.ndarray,
    entity_ids: List[int],
) -> np.ndarray:
    n_entities = len(entity_ids)
    max_entity_id = max(entity_ids)
    if embeddings.shape[0] == n_entities:
        return embeddings
    if embeddings.shape[0] > max_entity_id:
        return embeddings[np.asarray(entity_ids, dtype=np.int64)]
    raise ValueError(
        "Could not align embeddings to entities. "
        f"embeddings rows={embeddings.shape[0]}, entities={n_entities}, max_entity_id={max_entity_id}."
    )


def _maybe_array_list(obj: Any) -> list[np.ndarray] | None:
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        arrs: list[np.ndarray] = []
        for item in obj:
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim != 2:
                return None
            arrs.append(arr)
        return arrs
    return None


def extract_codebooks(rq: Any, expected_num_levels: int) -> list[np.ndarray]:
    """
    尝试从 RQKMeans 对象里提取每层 codebook。
    常见字段名都试一遍，尽量不依赖你内部实现的单一名字。
    """
    candidate_attr_names = [
        "codebooks",
        "_codebooks",
        "centroids",
        "_centroids",
        "codebook_centers",
        "cluster_centers_per_level",
        "centers_per_level",
    ]

    for attr_name in candidate_attr_names:
        if hasattr(rq, attr_name):
            maybe = _maybe_array_list(getattr(rq, attr_name))
            if maybe is not None and len(maybe) == expected_num_levels:
                return maybe

    # 兼容：每层对象里有 centroids / centers
    layer_container_names = [
        "layers",
        "_layers",
        "quantizers",
        "_quantizers",
        "level_models",
    ]
    for container_name in layer_container_names:
        if hasattr(rq, container_name):
            container = getattr(rq, container_name)
            if isinstance(container, (list, tuple)) and len(container) == expected_num_levels:
                codebooks: list[np.ndarray] = []
                ok = True
                for layer_obj in container:
                    found = None
                    for layer_attr in ["centroids", "centers", "cluster_centers", "codebook"]:
                        if hasattr(layer_obj, layer_attr):
                            arr = np.asarray(getattr(layer_obj, layer_attr), dtype=np.float32)
                            if arr.ndim == 2:
                                found = arr
                                break
                    if found is None:
                        ok = False
                        break
                    codebooks.append(found)
                if ok:
                    return codebooks

    raise RuntimeError(
        "Failed to extract codebooks from RQKMeans. "
        "Please inspect src/tkgr/rq_kmeans.py and expose per-level centroids, "
        "for example as rq.codebooks: List[np.ndarray] with shape [(K, D), ...]."
    )


def export_codebook_embeddings(
    codebooks: list[np.ndarray],
    output_dir: Path,
    code_pad_width: int,
) -> tuple[Path, Path, Path, list[str]]:
    """
    导出：
    1) codebook_embeddings.npz
    2) code_token_map.json
    3) code_token_embeddings.npy
    4) code_token_embeddings.tsv
    """
    npz_payload: dict[str, np.ndarray] = {}
    token_map: dict[str, dict[str, int]] = {}
    all_tokens: list[str] = []
    all_embeddings: list[np.ndarray] = []

    for level_idx, codebook in enumerate(codebooks, start=1):
        npz_payload[f"level_{level_idx}"] = np.asarray(codebook, dtype=np.float32)
        for code_idx in range(codebook.shape[0]):
            token = f"<z{level_idx}_{code_idx:0{code_pad_width}d}>"
            token_map[token] = {
                "level": level_idx,
                "code_index": code_idx,
            }
            all_tokens.append(token)
            all_embeddings.append(np.asarray(codebook[code_idx], dtype=np.float32))

    codebook_npz_path = output_dir / "codebook_embeddings.npz"
    np.savez(codebook_npz_path, **npz_payload)

    token_map_path = output_dir / "code_token_map.json"
    token_map_path.write_text(
        json.dumps(token_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    token_embedding_matrix = np.stack(all_embeddings, axis=0).astype(np.float32, copy=False)
    token_embedding_path = output_dir / "code_token_embeddings.npy"
    np.save(token_embedding_path, token_embedding_matrix)

    token_index_path = output_dir / "code_token_embeddings.tsv"
    with token_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_index", "token", "level", "code_index"])
        for row_idx, token in enumerate(all_tokens):
            meta = token_map[token]
            writer.writerow([row_idx, token, meta["level"], meta["code_index"]])

    return codebook_npz_path, token_map_path, token_embedding_path, all_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: build hierarchical semantic codes with RQ-KMeans and export code token prototypes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name recorded in the output summary.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing entity2id.txt.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Path to entity embedding matrix (.npy or .npz).",
    )
    parser.add_argument(
        "--npz-key",
        type=str,
        default=None,
        help="Optional key for .npz embedding files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for semantic-code outputs.",
    )
    parser.add_argument("--num-levels", type=int, default=4, help="Number of RQ levels.")
    parser.add_argument("--codebook-size", type=int, default=64, help="Clusters per level.")
    parser.add_argument(
        "--code-pad-width",
        type=int,
        default=2,
        help="Zero-padding width for code tokens, e.g. <z1_07>.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Training epochs per level.")
    parser.add_argument(
        "--max-steps-per-level",
        type=int,
        default=None,
        help="Optional absolute steps per level; overrides max-epochs.",
    )
    parser.add_argument(
        "--init-buffer-size",
        type=int,
        default=4096,
        help="Initialization buffer size used by K-Means++.",
    )
    parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=4096,
        help="Chunk size for distance computation to limit memory usage.",
    )
    parser.add_argument(
        "--no-normalize-residuals",
        action="store_true",
        help="Disable L2 normalization before each quantization layer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_path(args.dataset_dir, must_exist=True)
    embeddings_path = resolve_path(args.embeddings_path, must_exist=True)
    output_dir = resolve_path(args.output_dir, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity2id_path = dataset_dir / "entity2id.txt"
    if not entity2id_path.exists():
        raise FileNotFoundError(f"Missing file: {entity2id_path}")

    entity_records = load_entity2id(entity2id_path)
    entity_ids = [item[0] for item in entity_records]
    entity_names = [item[1] for item in entity_records]

    embeddings = load_embeddings(embeddings_path, args.npz_key)
    embeddings = align_embeddings_to_entities(embeddings, entity_ids)

    rq = RQKMeans(
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps_per_level=args.max_steps_per_level,
        init_buffer_size=args.init_buffer_size,
        normalize_residuals=not args.no_normalize_residuals,
        seed=args.seed,
        distance_chunk_size=args.distance_chunk_size,
    )

    codes = rq.fit_transform(embeddings)
    code_strings = codes_to_strings(codes)
    mapping = build_code_to_entity_index(entity_ids, code_strings)
    rq.save(output_dir)

    # 新增：导出每层 code token 的原型向量
    codebooks = extract_codebooks(rq, expected_num_levels=args.num_levels)
    codebook_npz_path, code_token_map_path, code_token_embeddings_path, all_code_tokens = export_codebook_embeddings(
        codebooks,
        output_dir,
        code_pad_width=args.code_pad_width,
    )

    entity_codes_path = output_dir / "entity_codes.tsv"
    with entity_codes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        header = ["entity_id", "entity_name"] + [f"z{i}" for i in range(args.num_levels)] + ["code"]
        writer.writerow(header)
        for idx, name, code_row, code_str in zip(entity_ids, entity_names, codes, code_strings):
            writer.writerow([idx, name] + [int(value) for value in code_row] + [code_str])

    code_to_entities_path = output_dir / "code_to_entities.json"
    code_to_entities_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "num_entities": len(entity_records),
        "dataset": args.dataset,
        "embedding_dim": int(embeddings.shape[1]),
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "code_pad_width": args.code_pad_width,
        "normalize_residuals": not args.no_normalize_residuals,
        "unique_code_rate": unique_code_rate(codes),
        "layer_metrics": [metric.__dict__ for metric in rq.layer_metrics],
        "num_code_tokens": len(all_code_tokens),
        "paths": {
            "dataset_dir": str(dataset_dir),
            "embeddings_path": str(embeddings_path),
            "output_dir": str(output_dir),
            "entity_codes_path": str(entity_codes_path),
            "code_to_entities_path": str(code_to_entities_path),
            "codebook_embeddings_path": str(codebook_npz_path),
            "code_token_map_path": str(code_token_map_path),
            "code_token_embeddings_path": str(code_token_embeddings_path),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Semantic-code generation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Entity codes: {entity_codes_path}")
    print(f"Codebook embeddings: {codebook_npz_path}")
    print(f"Code token map: {code_token_map_path}")
    print(f"Code token embeddings: {code_token_embeddings_path}")
    print(f"Summary: {summary_path}")
    print(f"Unique code rate: {summary['unique_code_rate']:.4f}")


if __name__ == "__main__":
    main()
