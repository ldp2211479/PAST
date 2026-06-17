#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Record = Tuple[int, str, str]


def resolve_path(path_str: str, *, must_exist: bool = False) -> Path:
    path = Path(path_str).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def load_description_csv(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        required = {"id", "name", "description"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"{path} must contain columns {sorted(required)}, missing {sorted(missing)}."
            )

        seen_ids: set[int] = set()
        for row in reader:
            clean_row = {key.strip(): value for key, value in row.items() if key is not None}
            idx = int(clean_row["id"])
            if idx in seen_ids:
                raise ValueError(f"Duplicate id={idx} in {path}.")
            seen_ids.add(idx)
            records.append(
                (
                    idx,
                    clean_row["name"].strip(),
                    clean_row["description"].strip(),
                )
            )

    records.sort(key=lambda item: item[0])
    return records


def build_input_text(name: str, description: str, instruction: str) -> str:
    text = f"{name}. {description}".strip()
    instruction = instruction.strip()
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def maybe_truncate_dim(embeddings: np.ndarray, embedding_dim: int | None) -> np.ndarray:
    if embedding_dim is None:
        return embeddings
    if embeddings.shape[1] < embedding_dim:
        raise ValueError(
            f"Model returned {embeddings.shape[1]} dims, smaller than --embedding-dim={embedding_dim}."
        )
    return embeddings[:, :embedding_dim]


def encode_with_sentence_transformers(
    texts: Sequence[str],
    *,
    model_path: Path,
    batch_size: int,
    normalize_embeddings: bool,
    embedding_dim: int | None,
    device: str | None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    init_kwargs = {"trust_remote_code": True}
    if device:
        init_kwargs["device"] = device

    model = SentenceTransformer(str(model_path), **init_kwargs)
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": normalize_embeddings,
    }
    encode_sig = inspect.signature(model.encode)
    if embedding_dim is not None and "truncate_dim" in encode_sig.parameters:
        encode_kwargs["truncate_dim"] = embedding_dim

    embeddings = model.encode(list(texts), **encode_kwargs)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = maybe_truncate_dim(embeddings, embedding_dim)
    return embeddings.astype(np.float32, copy=False)


def _last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def encode_with_transformers(
    texts: Sequence[str],
    *,
    model_path: Path,
    batch_size: int,
    normalize_embeddings: bool,
    embedding_dim: int | None,
    max_length: int,
    device: str | None,
) -> np.ndarray:
    try:
        from packaging.version import Version
    except Exception as exc:
        raise RuntimeError("Missing dependency packaging.") from exc

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if Version(transformers.__version__) < Version("4.51.0"):
        raise RuntimeError(
            "Qwen3-Embedding-4B requires transformers>=4.51.0. "
            f"Current transformers version is {transformers.__version__}."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    total = len(texts)
    with torch.inference_mode():
        for batch_id, batch_texts in enumerate(iter_batches(texts, batch_size), start=1):
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            embeddings = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if embedding_dim is not None:
                if embeddings.shape[1] < embedding_dim:
                    raise ValueError(
                        f"Model returned {embeddings.shape[1]} dims, smaller than --embedding-dim={embedding_dim}."
                    )
                embeddings = embeddings[:, :embedding_dim]
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())

            done = min(batch_id * batch_size, total)
            print(f"Encoded {done}/{total}", flush=True)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)


def encode_texts(
    texts: Sequence[str],
    *,
    model_path: Path,
    backend: str,
    batch_size: int,
    normalize_embeddings: bool,
    embedding_dim: int | None,
    max_length: int,
    device: str | None,
) -> tuple[np.ndarray, str]:
    if backend in ("auto", "sentence-transformers"):
        try:
            embeddings = encode_with_sentence_transformers(
                texts,
                model_path=model_path,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                embedding_dim=embedding_dim,
                device=device,
            )
            return embeddings, "sentence-transformers"
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            print(
                "sentence-transformers backend failed; falling back to transformers. "
                f"Reason: {type(exc).__name__}: {exc}",
                flush=True,
            )

    embeddings = encode_with_transformers(
        texts,
        model_path=model_path,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        embedding_dim=embedding_dim,
        max_length=max_length,
        device=device,
    )
    return embeddings, "transformers"


def write_index_tsv(path: Path, records: Sequence[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_index", "id", "name"])
        for row_index, (idx, name, _) in enumerate(records):
            writer.writerow([row_index, idx, name])


def build_embeddings_for_records(
    *,
    records: Sequence[Record],
    model_path: Path,
    output_path: Path,
    index_path: Path,
    instruction: str,
    backend: str,
    batch_size: int,
    normalize_embeddings: bool,
    embedding_dim: int | None,
    max_length: int,
    max_items: int | None,
    device: str | None,
) -> tuple[list[int], str]:
    selected_records = list(records if max_items is None else records[:max_items])
    texts = [
        build_input_text(name, description, instruction)
        for _, name, description in selected_records
    ]
    embeddings, used_backend = encode_texts(
        texts,
        model_path=model_path,
        backend=backend,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        embedding_dim=embedding_dim,
        max_length=max_length,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    write_index_tsv(index_path, selected_records)
    return list(embeddings.shape), used_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Qwen3 entity/relation embeddings from semantic description CSV files."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name recorded in the output summary.",
    )
    parser.add_argument(
        "--entity-descriptions",
        type=str,
        default=None,
        help="CSV with columns id,name,description for entities.",
    )
    parser.add_argument(
        "--relation-descriptions",
        type=str,
        default=None,
        help="CSV with columns id,name,description for relations.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local Qwen3-Embedding model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save embeddings and metadata.",
    )
    parser.add_argument(
        "--target",
        choices=["entity", "relation", "both"],
        default="entity",
        help="Which descriptions to encode.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "transformers"],
        default="auto",
        help="Embedding backend. auto tries sentence-transformers first, then transformers.",
    )
    parser.add_argument(
        "--entity-instruction",
        type=str,
        default="Represent the entity description for temporal knowledge graph reasoning.",
    )
    parser.add_argument(
        "--relation-instruction",
        type=str,
        default="Represent the relation description for temporal knowledge graph reasoning.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional output dimension. Qwen3-Embedding-4B supports truncation up to 2560.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization before saving embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device passed to the backend, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional debug limit for the first N sorted rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model_path, must_exist=True)
    entity_descriptions_arg = args.entity_descriptions
    relation_descriptions_arg = args.relation_descriptions
    output_dir_arg = args.output_dir
    if output_dir_arg is None:
        raise ValueError("Please pass --output-dir.")
    if args.target in ("entity", "both") and entity_descriptions_arg is None:
        raise ValueError("Please pass --entity-descriptions.")
    if args.target in ("relation", "both") and relation_descriptions_arg is None:
        raise ValueError("Please pass --relation-descriptions.")

    output_dir = resolve_path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize_embeddings = not args.no_normalize_embeddings
    summary: dict[str, object] = {
        "model_path": str(model_path),
        "dataset": args.dataset,
        "target": args.target,
        "backend_requested": args.backend,
        "normalize_embeddings": normalize_embeddings,
        "embedding_dim": args.embedding_dim,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "paths": {
            "output_dir": str(output_dir),
        },
    }

    used_backends: dict[str, str] = {}

    if args.target in ("entity", "both"):
        entity_descriptions = resolve_path(entity_descriptions_arg, must_exist=True)
        entity_records = load_description_csv(entity_descriptions)
        shape, used_backend = build_embeddings_for_records(
            records=entity_records,
            model_path=model_path,
            output_path=output_dir / "entity_embeddings.npy",
            index_path=output_dir / "entity_embedding_index.tsv",
            instruction=args.entity_instruction,
            backend=args.backend,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            embedding_dim=args.embedding_dim,
            max_length=args.max_length,
            max_items=args.max_items,
            device=args.device,
        )
        summary["entity_embeddings_shape"] = shape
        summary["entity_descriptions"] = str(entity_descriptions)
        used_backends["entity"] = used_backend

    if args.target in ("relation", "both"):
        relation_descriptions = resolve_path(relation_descriptions_arg, must_exist=True)
        relation_records = load_description_csv(relation_descriptions)
        shape, used_backend = build_embeddings_for_records(
            records=relation_records,
            model_path=model_path,
            output_path=output_dir / "relation_embeddings.npy",
            index_path=output_dir / "relation_embedding_index.tsv",
            instruction=args.relation_instruction,
            backend=args.backend,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            embedding_dim=args.embedding_dim,
            max_length=args.max_length,
            max_items=args.max_items,
            device=args.device,
        )
        summary["relation_embeddings_shape"] = shape
        summary["relation_descriptions"] = str(relation_descriptions)
        used_backends["relation"] = used_backend

    summary["backend_used"] = used_backends
    summary["paths"] = {
        **summary["paths"],
        "entity_embeddings": str(output_dir / "entity_embeddings.npy"),
        "entity_embedding_index": str(output_dir / "entity_embedding_index.tsv"),
        "relation_embeddings": str(output_dir / "relation_embeddings.npy"),
        "relation_embedding_index": str(output_dir / "relation_embedding_index.tsv"),
    }

    summary_path = output_dir / "summary_embeddings.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Embedding generation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
