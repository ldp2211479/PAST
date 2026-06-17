#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

Record = Tuple[int, str]
Quadruple = Tuple[int, int, int, int]  # (subject, relation, object, timestamp)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def resolve_cli_path(path_like: str) -> Path:
    raw = str(path_like).strip()
    if not raw:
        raise ValueError("Path argument cannot be empty.")
    return Path(raw).expanduser()


def display_path(path: Path) -> str:
    return Path(path).expanduser().as_posix()


def load_id_name_tsv(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Invalid id mapping line in {path}: {line}")
            name, idx = parts[:2]
            records.append((int(idx), name))
    records.sort(key=lambda x: x[0])
    return records


def load_quadruples(path: Path) -> List[Quadruple]:
    rows: List[Quadruple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            subject, relation, obj, timestamp = parts[:4]
            rows.append((int(subject), int(relation), int(obj), int(timestamp)))
    return rows


def normalize_label(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "<" and text[-1] == ">":
        text = text[1:-1]
    return " ".join(text.replace("_", " ").replace("\t", " ").split())


def build_history_context_indexes(
    quadruples: List[Quadruple],
    entity_names: Dict[int, str],
    relation_names: Dict[int, str],
) -> Tuple[Dict[int, Counter[str]], Dict[int, Counter[str]]]:
    entity_context_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    relation_context_counts: Dict[int, Counter[str]] = defaultdict(Counter)

    for subject, relation, obj, _ in quadruples:
        subject_name = normalize_label(entity_names.get(subject, str(subject)))
        object_name = normalize_label(entity_names.get(obj, str(obj)))
        relation_name = normalize_label(relation_names.get(relation, str(relation)))

        entity_context_counts[subject][
            f"OUT relation={relation_name}; counterpart={object_name}"
        ] += 1
        entity_context_counts[obj][
            f"IN relation={relation_name}; counterpart={subject_name}"
        ] += 1
        relation_context_counts[relation][f"{subject_name} -> {object_name}"] += 1

    return entity_context_counts, relation_context_counts


def build_context_lookup(
    records: List[Record],
    context_counts: Dict[int, Counter[str]],
    max_context_facts: int,
    min_context_frequency: int,
) -> Dict[int, List[str]]:
    lookup: Dict[int, List[str]] = {}
    for idx, _ in records:
        counter = context_counts.get(idx)
        if not counter:
            continue
        facts: List[str] = []
        for text, count in counter.most_common():
            if count < min_context_frequency:
                continue
            facts.append(f"{text}; freq={count}")
            if len(facts) >= max_context_facts:
                break
        if facts:
            lookup[idx] = facts
    return lookup

def sniff_delimiter(path: Path) -> str:
    """
    自动识别分隔符。优先识别逗号和制表符。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except Exception:
        if "\t" in sample and "," not in sample:
            return "\t"
        return ","


def load_records_from_csv(path: Path) -> List[Record]:
    """
    从 csv/tsv 中读取 (id, name)。
    要求至少包含 id, name 两列。
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    delimiter = sniff_delimiter(path)
    records: List[Record] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        normalized_fields = {field.strip().lower(): field for field in reader.fieldnames}
        if "id" not in normalized_fields or "name" not in normalized_fields:
            raise ValueError(
                f"CSV file must contain 'id' and 'name' columns, got: {reader.fieldnames}"
            )

        id_col = normalized_fields["id"]
        name_col = normalized_fields["name"]

        for row in reader:
            raw_id = str(row.get(id_col, "")).strip()
            raw_name = str(row.get(name_col, "")).strip()
            if not raw_id or not raw_name:
                continue
            records.append((int(raw_id), raw_name))

    records.sort(key=lambda x: x[0])
    return records


def read_existing_descriptions_from_csv(path: Path) -> Dict[int, Dict[str, str]]:
    """
    从 csv/tsv 中读取已有 description。
    如果没有 description 列，会自动视为空。
    """
    if not path.exists():
        return {}

    delimiter = sniff_delimiter(path)
    cache: Dict[int, Dict[str, str]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return {}

        normalized_fields = {field.strip().lower(): field for field in reader.fieldnames}
        id_col = normalized_fields.get("id")
        name_col = normalized_fields.get("name")
        desc_col = normalized_fields.get("description")

        if id_col is None or name_col is None:
            raise ValueError(
                f"CSV file must contain 'id' and 'name' columns, got: {reader.fieldnames}"
            )

        for row in reader:
            raw_id = str(row.get(id_col, "")).strip()
            if not raw_id:
                continue
            idx = int(raw_id)
            cache[idx] = {
                "name": str(row.get(name_col, "")).strip(),
                "description": str(row.get(desc_col, "")).strip() if desc_col else "",
            }
    return cache


def write_descriptions_to_csv(
    path: Path,
    id_name_records: List[Record],
    rows: Dict[int, Dict[str, str]],
) -> None:
    """
    直接写回 csv 文件：
    列固定为 id,name,description
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name", "description"])
        for idx, name in id_name_records:
            description = rows.get(idx, {}).get("description", "")
            writer.writerow([idx, name, description])


def normalize_description(text: str, max_chars: int, max_sentences: int = 4) -> str:
    cleaned = " ".join(text.replace("\n", " ").replace("\t", " ").split())
    sentence_parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", cleaned)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

    if sentence_parts:
        cleaned = " ".join(sentence_parts[:max_sentences]).strip()

    if cleaned and cleaned[-1] not in ".!?\u3002\uff01\uff1f":
        cleaned = f"{cleaned}."

    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."

    return cleaned


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def create_prompt(
    kind: str,
    name: str,
    language: str,
    context_facts: List[str],
    max_words: int,
    allow_world_knowledge: bool = True,
    min_sentences: int = 2,
    max_sentences: int = 4,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You generate accurate, informative, and readable descriptions for entities and relations "
        "in a temporal knowledge graph. "
        "Ground the description primarily in the provided historical evidence. "
        "When the item is widely known, you may incorporate relevant real-world knowledge to improve "
        "completeness and clarity, but do not speculate or invent uncertain facts. "
        "Always write the final answer in English only."
    )

    if context_facts:
        evidence_block = "\n".join(f"- {item}" for item in context_facts)
        evidence_note = "Historical evidence is available and should be the primary grounding source."
    else:
        evidence_block = "- NONE"
        evidence_note = (
            "Historical evidence is sparse or unavailable. In this case, stay conservative and only use "
            "high-confidence world knowledge if the name is widely recognizable."
        )

    if kind == "entity":
        task_block = (
            f"Write a coherent and informative description of this entity in {language}.\n"
            "The description should explain:\n"
            "1) what the entity is (type/category if inferable),\n"
            "2) its main role or significance,\n"
            "3) how it typically interacts with other entities in the temporal knowledge graph,\n"
            "4) and, if supported, any historical or temporal interaction pattern.\n"
        )
    else:
        task_block = (
            f"Write a coherent and informative description of this relation in {language}.\n"
            "The description should explain:\n"
            "1) the core meaning of the relation,\n"
            "2) what kinds of subjects usually appear on the source side,\n"
            "3) what kinds of objects usually appear on the target side,\n"
            "4) and, if supported, any recurring historical usage pattern.\n"
        )

    world_knowledge_rule = (
        "You may incorporate relevant world knowledge when it is high-confidence and helps make the "
        "description more accurate and complete."
        if allow_world_knowledge
        else
        "Do not add any world knowledge beyond what can be inferred directly from the provided evidence."
    )

    user_prompt = (
        f"{task_block}\n"
        f"{kind.capitalize()} name: {name}\n"
        "Historical evidence from the temporal knowledge graph:\n"
        f"{evidence_block}\n\n"
        f"Guidance: {evidence_note}\n\n"
        "Requirements:\n"
        f"1) Write {min_sentences} to {max_sentences} complete sentences, with about 80-{max_words} words in total.\n"
        "2) The output language must be English only. Do not write Chinese or any non-English text.\n"
        "3) Use a natural, encyclopedic, information-rich style.\n"
        "4) First clearly identify what the entity is in the real world (e.g., country, person, organization, and if possible its region or role).\n"
        "5) Provide a concise background description using your world knowledge if the entity is recognizable.\n"
        "6) Then describe how it behaves in the temporal knowledge graph based on the provided evidence.\n"
        f"7) {world_knowledge_rule}\n"
        "8) Do NOT mention frequencies, IDs, timestamps, or that evidence was provided.\n"
        "9) Do NOT use bullet points, numbering, or quotation marks.\n"
        "10) If evidence is sparse, stay conservative and avoid overclaiming.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url=api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        content = response.read().decode("utf-8")
    obj = json.loads(content)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices found in API response: {obj}")
    message = choices[0].get("message") or {}
    text = message.get("content", "")
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = " ".join(parts)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Empty message content in API response: {obj}")
    return text


def generate_with_retry(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
    require_english: bool,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            text = chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
            if require_english and contains_cjk(text):
                raise RuntimeError("Model returned non-English/CJK text while English-only output is required.")
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_time = retry_backoff_sec * (2 ** attempt)
            time.sleep(sleep_time)
    raise RuntimeError(f"Description generation failed after retries: {last_error}") from last_error


def generate_description_for_record(
    *,
    kind: str,
    idx: int,
    name: str,
    api_base: str,
    api_key: str,
    model: str,
    language: str,
    context_lookup: Dict[int, List[str]],
    max_words: int,
    allow_world_knowledge: bool,
    min_sentences: int,
    max_sentences: int,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
    sleep_sec: float,
    max_chars: int,
    require_english: bool,
) -> Tuple[int, str, str]:
    messages = create_prompt(
        kind=kind,
        name=name,
        language=language,
        context_facts=context_lookup.get(idx, []),
        max_words=max_words,
        allow_world_knowledge=allow_world_knowledge,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )

    text = generate_with_retry(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        require_english=require_english,
    )

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    return (
        idx,
        name,
        normalize_description(
            text,
            max_chars=max_chars,
            max_sentences=4,
        ),
    )


def run_generation(
    *,
    kind: str,
    records: List[Record],
    csv_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    language: str,
    context_lookup: Dict[int, List[str]],
    max_words: int,
    allow_world_knowledge: bool,
    min_sentences: int,
    max_sentences: int,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
    sleep_sec: float,
    max_chars: int,
    save_every: int,
    max_items: int | None,
    force_regenerate: bool,
    num_workers: int,
    require_english: bool,
) -> Tuple[int, int]:
    """
    核心改动：
    - 直接把 csv_path 当作输入+输出文件
    - 读取已有 description
    - 跳过已经生成的
    - 每生成一条立刻写回 csv
    """
    rows = read_existing_descriptions_from_csv(csv_path)
    generated = 0
    skipped = 0

    targets = records if max_items is None else records[:max_items]
    pending_targets: List[Record] = []
    for idx, name in targets:
        existing_desc = rows.get(idx, {}).get("description", "").strip()
        if (not force_regenerate) and existing_desc:
            skipped += 1
            continue
        pending_targets.append((idx, name))

    progress = None
    if tqdm is not None:
        progress = tqdm(pending_targets, total=len(pending_targets), desc=f"{kind} descriptions", unit="item")
        iterator = progress
    else:
        iterator = pending_targets
        print(
            f"[{kind}] Processing {len(pending_targets)} items "
            f"({skipped} skipped; install `tqdm` for a progress bar)."
        )

    def store_result(idx: int, name: str, description: str) -> None:
        nonlocal generated
        rows[idx] = {
            "name": name,
            "description": description,
        }
        generated += 1

        # 每生成一条立即写回，确保中断后可以继续
        write_descriptions_to_csv(csv_path, records, rows)

        if progress is None and (generated % max(1, save_every) == 0):
            print(f"[{kind}] Generated {generated} descriptions... skipped {skipped} existing rows.")

    worker_count = max(1, int(num_workers))
    if worker_count == 1:
        for idx, name in iterator:
            result_idx, result_name, description = generate_description_for_record(
                kind=kind,
                idx=idx,
                name=name,
                api_base=api_base,
                api_key=api_key,
                model=model,
                language=language,
                context_lookup=context_lookup,
                max_words=max_words,
                allow_world_knowledge=allow_world_knowledge,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
                sleep_sec=sleep_sec,
                max_chars=max_chars,
                require_english=require_english,
            )
            store_result(result_idx, result_name, description)
    else:
        if progress is None:
            print(f"[{kind}] Using {worker_count} concurrent workers.")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    generate_description_for_record,
                    kind=kind,
                    idx=idx,
                    name=name,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    language=language,
                    context_lookup=context_lookup,
                    max_words=max_words,
                    allow_world_knowledge=allow_world_knowledge,
                    min_sentences=min_sentences,
                    max_sentences=max_sentences,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                    retry_backoff_sec=retry_backoff_sec,
                    sleep_sec=sleep_sec,
                    max_chars=max_chars,
                    require_english=require_english,
                )
                for idx, name in pending_targets
            ]
            for future in as_completed(futures):
                result_idx, result_name, description = future.result()
                store_result(result_idx, result_name, description)
                if progress is not None:
                    progress.update(1)

    write_descriptions_to_csv(csv_path, records, rows)
    if progress is not None:
        progress.close()

    filled = sum(1 for idx, _ in records if rows.get(idx, {}).get("description", "").strip())
    return generated, filled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0-A: generate short entity/relation descriptions via GPT chat completions API."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name recorded in the run summary.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset folder containing entity2id.txt and relation2id.txt.",
    )

    parser.add_argument(
        "--entity-csv",
        type=str,
        default=None,
        help="Entity CSV file with columns: id,name,description. The script resumes generation directly in this file.",
    )
    parser.add_argument(
        "--relation-csv",
        type=str,
        default=None,
        help="Relation CSV file with columns: id,name,description. The script resumes generation directly in this file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated description files when CSV files are not provided.",
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
        help="Chat completions endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key string. If omitted, reads from --api-key-env.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name used when --api-key is omitted.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model name.")
    parser.add_argument(
        "--target",
        type=str,
        choices=["entity", "relation", "both"],
        default="both",
        help="Which descriptions to generate.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Description language hint passed to model.",
    )
    parser.add_argument(
        "--context-split",
        type=str,
        default="train.txt",
        help="Dataset split file used to build high-frequency historical context.",
    )
    parser.add_argument(
        "--max-context-facts",
        type=int,
        default=50,
        help="Maximum number of high-frequency context facts injected per entity/relation.",
    )
    parser.add_argument(
        "--min-context-frequency",
        type=int,
        default=1,
        help="Minimum frequency threshold for a context fact to be included.",
    )
    parser.add_argument(
        "--disable-history-context",
        action="store_true",
        help="Disable high-frequency historical context injection.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=4096,
        help="Approximate upper bound for the full description length in words.",
    )
    parser.add_argument(
        "--allow-world-knowledge",
        action="store_true",
        default=True,
        help="Allow the model to incorporate high-confidence world knowledge beyond graph evidence.",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=4,
        help="Minimum number of sentences to request.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="Maximum number of sentences to request.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--max-chars", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Number of concurrent API requests. Use 1 for sequential generation.",
    )
    parser.add_argument(
        "--allow-non-english",
        action="store_true",
        help="Disable English-only output validation.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Debug mode: limit items per target.")
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"No API key found. Provide --api-key or set env variable {args.api_key_env}."
        )

    dataset_dir = resolve_cli_path(args.dataset_dir)
    output_dir = resolve_cli_path(args.output_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_path = dataset_dir / "entity2id.txt"
    relation_path = dataset_dir / "relation2id.txt"
    if not entity_path.exists():
        raise FileNotFoundError(f"Missing file: {dataset_dir / 'entity2id.txt'}")
    if not relation_path.exists():
        raise FileNotFoundError(f"Missing file: {dataset_dir / 'relation2id.txt'}")

    if args.entity_csv:
        entity_csv_path = resolve_cli_path(args.entity_csv)
        entity_records = load_records_from_csv(entity_csv_path)
    else:
        entity_csv_path = output_dir / "entity_descriptions.csv"
        entity_records = load_id_name_tsv(entity_path)
        if not entity_csv_path.exists():
            init_rows = {idx: {"name": name, "description": ""} for idx, name in entity_records}
            write_descriptions_to_csv(entity_csv_path, entity_records, init_rows)

    if args.relation_csv:
        relation_csv_path = resolve_cli_path(args.relation_csv)
        relation_records = load_records_from_csv(relation_csv_path)
    else:
        relation_csv_path = output_dir / "relation_descriptions.csv"
        relation_records = load_id_name_tsv(relation_path)
        if not relation_csv_path.exists():
            init_rows = {idx: {"name": name, "description": ""} for idx, name in relation_records}
            write_descriptions_to_csv(relation_csv_path, relation_records, init_rows)

    entity_name_map = {idx: name for idx, name in entity_records}
    relation_name_map = {idx: name for idx, name in relation_records}

    if args.disable_history_context:
        context_path = None
        context_quadruples: List[Quadruple] = []
        entity_context_lookup: Dict[int, List[str]] = {}
        relation_context_lookup: Dict[int, List[str]] = {}
    else:
        raw_context_path = Path(args.context_split)
        context_path = (
            raw_context_path
            if raw_context_path.is_absolute()
            else dataset_dir / raw_context_path
        )
        if not context_path.exists():
            raise FileNotFoundError(
                f"Missing context split file: {display_path(context_path)}"
            )
        context_quadruples = load_quadruples(context_path)
        entity_context_counts, relation_context_counts = build_history_context_indexes(
            context_quadruples,
            entity_names=entity_name_map,
            relation_names=relation_name_map,
        )
        entity_context_lookup = build_context_lookup(
            records=entity_records,
            context_counts=entity_context_counts,
            max_context_facts=args.max_context_facts,
            min_context_frequency=args.min_context_frequency,
        )
        relation_context_lookup = build_context_lookup(
            records=relation_records,
            context_counts=relation_context_counts,
            max_context_facts=args.max_context_facts,
            min_context_frequency=args.min_context_frequency,
        )

    entity_generated = 0
    relation_generated = 0
    entity_filled = 0
    relation_filled = 0

    if args.target in ("entity", "both"):
        entity_generated, entity_filled = run_generation(
            kind="entity",
            records=entity_records,
            csv_path=entity_csv_path,
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            language=args.language,
            context_lookup=entity_context_lookup,
            max_words=args.max_words,
            allow_world_knowledge=args.allow_world_knowledge,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            retry_backoff_sec=args.retry_backoff_sec,
            sleep_sec=args.sleep_sec,
            max_chars=args.max_chars,
            save_every=args.save_every,
            max_items=args.max_items,
            force_regenerate=args.force_regenerate,
            num_workers=args.num_workers,
            require_english=not args.allow_non_english,
        )

    if args.target in ("relation", "both"):
        relation_generated, relation_filled = run_generation(
            kind="relation",
            records=relation_records,
            csv_path=relation_csv_path,
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            language=args.language,
            context_lookup=relation_context_lookup,
            max_words=args.max_words,
            allow_world_knowledge=args.allow_world_knowledge,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            retry_backoff_sec=args.retry_backoff_sec,
            sleep_sec=args.sleep_sec,
            max_chars=args.max_chars,
            save_every=args.save_every,
            max_items=args.max_items,
            force_regenerate=args.force_regenerate,
            num_workers=args.num_workers,
            require_english=not args.allow_non_english,
        )

    summary = {
        "model": args.model,
        "api_base": args.api_base,
        "target": args.target,
        "language": args.language,
        "english_only_validation": not args.allow_non_english,
        "history_context_enabled": not args.disable_history_context,
        "context_split": (
            display_path(context_path)
            if context_path is not None
            else None
        ),
        "context_quadruples": len(context_quadruples),
        "max_context_facts": args.max_context_facts,
        "min_context_frequency": args.min_context_frequency,
        "max_words": args.max_words,
        "num_workers": args.num_workers,
        "entity_total": len(entity_records),
        "relation_total": len(relation_records),
        "entity_generated_this_run": entity_generated,
        "relation_generated_this_run": relation_generated,
        "entity_descriptions_filled": entity_filled,
        "relation_descriptions_filled": relation_filled,
        "entity_context_covered": sum(
            1 for idx, _ in entity_records if idx in entity_context_lookup
        ),
        "relation_context_covered": sum(
            1 for idx, _ in relation_records if idx in relation_context_lookup
        ),
        "paths": {
            "dataset_dir": display_path(dataset_dir),
            "output_dir": display_path(output_dir),
            "entity_csv_path": display_path(entity_csv_path),
            "relation_csv_path": display_path(relation_csv_path),
        },
    }

    summary_path = output_dir / "summary_descriptions.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Description generation completed.")
    print(f"Output directory: {display_path(output_dir)}")
    print(f"Entity CSV: {display_path(entity_csv_path)}")
    print(f"Relation CSV: {display_path(relation_csv_path)}")
    print(f"Summary: {display_path(summary_path)}")


if __name__ == "__main__":
    main()
