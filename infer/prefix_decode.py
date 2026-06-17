#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
END_KEY = "__END__"
CODE_TOKEN_RE = re.compile(r"<[^>]+>")
LLM = Any
SamplingParams = None
LoRARequest = None


@dataclass
class Beam:
    token_ids: List[int]
    code_tokens: List[Any]
    score: float
    llm_score: float
    graph_score: float


SCORE_MODES = ("hybrid", "llm", "graph")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "infer.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def sanitized_args(args: argparse.Namespace) -> Dict[str, Any]:
    path_keys = {
        "model_path",
        "lora_path",
        "tokenizer_path",
        "prompts_jsonl",
        "graph_score_dir",
        "entity_codes",
        "entity_code_map",
        "entity_trie_pkl",
        "code_token_index_path",
        "entity2id",
        "output_dir",
    }
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in path_keys and value:
            out[key] = "<path>"
        else:
            out[key] = value
    return out


def resolve_path(path: str, *, must_exist: bool = False) -> Path:
    value = Path(path).expanduser()
    if must_exist and not value.exists():
        raise FileNotFoundError(f"Path does not exist: {value}")
    return value


def load_vllm_runtime() -> None:
    global LLM, SamplingParams, LoRARequest
    try:
        from vllm import LLM as VLLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required for inference. Please install vllm before decoding.") from exc

    LLM = VLLM
    SamplingParams = VLLMSamplingParams
    try:
        from vllm.lora.request import LoRARequest as VLLMLoRARequest
    except Exception:
        VLLMLoRARequest = None
    LoRARequest = VLLMLoRARequest


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(item)
    return rows


def read_samples(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Expected JSON array of objects in {path}")
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "records"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    raise ValueError(f"Unsupported sample file format: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -float("inf")
    max_v = max(values)
    if math.isinf(max_v):
        return max_v
    return max_v + math.log(sum(math.exp(float(v) - max_v) for v in values))


def log_softmax_dict(scores: Dict[Any, float]) -> Dict[Any, float]:
    if not scores:
        return {}
    denom = logsumexp(list(scores.values()))
    if math.isinf(denom):
        uniform = -math.log(len(scores))
        return {key: uniform for key in scores}
    return {key: float(value) - denom for key, value in scores.items()}


def softmax_prob_dict(scores: Dict[Any, float]) -> Dict[Any, float]:
    return {key: math.exp(value) for key, value in log_softmax_dict(scores).items()}


def minmax_softmax_prob_dict(scores: Dict[Any, float]) -> Dict[Any, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        scaled = {key: 1.0 for key in scores}
    else:
        scaled = {key: (float(value) - lo) / (hi - lo) for key, value in scores.items()}
    return softmax_prob_dict(scaled)


def normalize_prob_dict(scores: Dict[Any, float], method: str) -> Dict[Any, float]:
    clean = {key: float(value) for key, value in scores.items() if not math.isinf(float(value))}
    if not clean:
        return {}
    if method == "softmax":
        return softmax_prob_dict(clean)
    if method == "minmax_softmax":
        return minmax_softmax_prob_dict(clean)
    if method == "none":
        total = sum(max(value, 0.0) for value in clean.values())
        if total <= 0.0:
            return {key: 1.0 / len(clean) for key in clean}
        return {key: max(value, 0.0) / total for key, value in clean.items()}
    raise ValueError(f"Unknown normalization method: {method}")


def load_entity_names(path: Optional[Path]) -> Dict[int, str]:
    if path is None or not path.exists():
        return {}
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            name, idx = parts
            out[int(idx)] = name.replace("_", " ")
    return out


def load_entity_codes(
    path: Path,
    *,
    code_token_index_path: Optional[Path],
    num_levels: int,
) -> Dict[int, Tuple[str, ...]]:
    if code_token_index_path is None:
        sibling_index = path.with_name("code_token_embeddings.tsv")
        if sibling_index.exists():
            code_token_index_path = sibling_index
    if code_token_index_path is None or not code_token_index_path.exists():
        raise FileNotFoundError(
            f"Expected code token index TSV from rq_kmeans: {path.with_name('code_token_embeddings.tsv')}"
        )

    token_by_level_code: Dict[Tuple[int, int], str] = {}
    with code_token_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in code token index file: {code_token_index_path}")
        required = {"token", "level", "code_index"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{code_token_index_path} missing columns: {sorted(missing)}")
        for row in reader:
            token_by_level_code[(int(row["level"]), int(row["code_index"]))] = row["token"].strip()

    out: Dict[int, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in entity code file: {path}")
        z_cols = [f"z{i}" for i in range(num_levels)]
        if not all(col in reader.fieldnames for col in z_cols):
            alt_cols = [f"z{i + 1}" for i in range(num_levels)]
            if all(col in reader.fieldnames for col in alt_cols):
                z_cols = alt_cols
            else:
                raise ValueError(f"{path} must contain {z_cols} or {alt_cols}")
        for row in reader:
            entity_id = int(row["entity_id"])
            tokens = []
            for level, col in enumerate(z_cols, start=1):
                code_id = int(row[col])
                token = token_by_level_code.get((level, code_id))
                if token is None:
                    raise ValueError(
                        f"No token found in {code_token_index_path} for entity_id={entity_id}, "
                        f"level={level}, code_index={code_id}."
                    )
                tokens.append(token)
            tokens = tuple(tokens)
            out[entity_id] = tokens
    if not out:
        raise ValueError(f"No entity codes loaded from {path}")
    return out


def split_code_tokens(code_text: str) -> Tuple[str, ...]:
    tokens = tuple(CODE_TOKEN_RE.findall(code_text))
    if tokens:
        return tokens
    parts = tuple(part for part in code_text.strip().split() if part)
    if parts:
        return parts
    raise ValueError(f"Could not parse semantic code string: {code_text!r}")


def load_entity_code_map(path: Path, *, num_levels: int) -> Dict[int, Tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected entity code map JSON object: {path}")

    out: Dict[int, Tuple[str, ...]] = {}
    for raw_entity_id, raw_code in payload.items():
        entity_id = int(raw_entity_id)
        tokens = split_code_tokens(str(raw_code))
        if len(tokens) != int(num_levels):
            raise ValueError(
                f"Entity {entity_id} has {len(tokens)} code tokens, expected {num_levels}: {raw_code}"
            )
        out[entity_id] = tokens
    if not out:
        raise ValueError(f"No entity codes loaded from {path}")
    return out


def build_code_to_entities(entity_codes: Dict[int, Tuple[str, ...]]) -> Dict[Tuple[str, ...], List[int]]:
    out: Dict[Tuple[str, ...], List[int]] = {}
    for entity_id, code in entity_codes.items():
        out.setdefault(tuple(code), []).append(int(entity_id))
    return out


def tokenize_entity_codes(
    entity_codes: Dict[int, Tuple[str, ...]],
    tokenizer: Any,
) -> tuple[Dict[int, Tuple[int, ...]], Dict[int, str]]:
    token_id_to_token: Dict[int, str] = {}
    entity_code_ids: Dict[int, Tuple[int, ...]] = {}
    for entity_id, code in entity_codes.items():
        token_ids: List[int] = []
        for token in code:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or int(token_id) < 0:
                raise ValueError(f"Tokenizer does not contain semantic-code token: {token}")
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1 or int(encoded[0]) != int(token_id):
                raise ValueError(
                    f"Semantic-code token must be atomic in tokenizer: {token}, encoded={encoded}"
                )
            token_id = int(token_id)
            token_ids.append(token_id)
            token_id_to_token[token_id] = token
        entity_code_ids[entity_id] = tuple(token_ids)
    return entity_code_ids, token_id_to_token


class SemanticCodeTrie:
    def __init__(self, entity_codes: Dict[int, Tuple[str, ...]], token_to_id: Dict[str, int]):
        self.entity_codes = entity_codes
        self.token_to_id = token_to_id
        self.root: Dict[str, Any] = {"children": {}, "entities": []}
        self.code_to_entities: Dict[Tuple[str, ...], List[int]] = {}
        self.max_depth = 0
        for entity_id, code in entity_codes.items():
            self.max_depth = max(self.max_depth, len(code))
            self.code_to_entities.setdefault(code, []).append(entity_id)
            node = self.root
            node["entities"].append(entity_id)
            for token in code:
                if token not in token_to_id:
                    raise ValueError(
                        f"Semantic code token {token!r} is not a single tokenizer token. "
                        "Load the warmed/LoRA tokenizer that contains the added semantic-code tokens."
                    )
                node = node["children"].setdefault(token, {"children": {}, "entities": []})
                node["entities"].append(entity_id)

    def node_for_prefix(self, prefix: Sequence[str]) -> Optional[Dict[str, Any]]:
        node = self.root
        for token in prefix:
            node = node["children"].get(token)
            if node is None:
                return None
        return node

    def allowed_tokens(self, prefix: Sequence[str]) -> List[str]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return []
        return sorted(node["children"].keys())

    def entities_for_prefix(self, prefix: Sequence[str]) -> List[int]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return []
        return [int(x) for x in node["entities"]]

    def entities_for_code(self, code: Sequence[str]) -> List[int]:
        return self.code_to_entities.get(tuple(code), [])


class PrebuiltTokenIdTrie:
    def __init__(self, trie: Dict[Any, Any], entity_code_ids: Dict[int, Tuple[int, ...]]):
        if not isinstance(trie, dict):
            raise ValueError("Prebuilt entity trie must be a dictionary.")
        self.root = trie
        self.entity_codes = entity_code_ids
        self.code_to_entities: Dict[Tuple[int, ...], List[int]] = {}
        self.prefix_to_entities: Dict[Tuple[int, ...], List[int]] = {}
        self.max_depth = 0
        for entity_id, code in entity_code_ids.items():
            self.max_depth = max(self.max_depth, len(code))
            self.code_to_entities.setdefault(tuple(code), []).append(int(entity_id))
            for depth in range(len(code) + 1):
                self.prefix_to_entities.setdefault(tuple(code[:depth]), []).append(int(entity_id))

    def node_for_prefix(self, prefix: Sequence[int]) -> Optional[Dict[Any, Any]]:
        node: Dict[Any, Any] = self.root
        for token_id in prefix:
            node = node.get(int(token_id))
            if node is None:
                return None
        return node

    def allowed_tokens(self, prefix: Sequence[int]) -> List[int]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return []
        return sorted(int(key) for key in node.keys() if key != END_KEY)

    def entities_for_prefix(self, prefix: Sequence[int]) -> List[int]:
        return self.prefix_to_entities.get(tuple(int(x) for x in prefix), [])

    def entities_for_code(self, code: Sequence[int]) -> List[int]:
        return self.code_to_entities.get(tuple(int(x) for x in code), [])


class DenseGraphScoreCache:
    def __init__(self, score_dir: Path, split: str):
        self.score_dir = score_dir
        self.split = split
        self.score_path = score_dir / f"{split}_graph_scores.npy"
        self.query_path = score_dir / f"{split}_graph_queries.jsonl"
        self.summary_path = score_dir / f"{split}_graph_scores_summary.json"
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.scores = np.load(self.score_path, mmap_mode="r")
        self.queries = read_jsonl(self.query_path)
        self.id_to_row = {str(item["id"]): int(item["row_index"]) for item in self.queries}
        self._validate()

    def _validate(self) -> None:
        expected = (int(self.summary["num_queries"]), int(self.summary["num_entities"]))
        if tuple(self.scores.shape) != expected:
            raise ValueError(f"Graph score shape mismatch: got={self.scores.shape}, expected={expected}")
        if len(self.queries) != expected[0]:
            raise ValueError(f"Query metadata count mismatch: got={len(self.queries)}, expected={expected[0]}")

    def get(self, query_id: str) -> np.ndarray:
        row = self.id_to_row.get(str(query_id))
        if row is None:
            try:
                candidate = int(query_id)
            except (TypeError, ValueError):
                candidate = -1
            if 0 <= candidate < int(self.scores.shape[0]):
                row = candidate
        if row is None:
            raise KeyError(f"Query id not found in graph cache: {query_id}")
        return np.asarray(self.scores[row], dtype=np.float32)


def extract_step_topk_logprobs(vllm_output: Any) -> Dict[int, float]:
    if vllm_output is None or not getattr(vllm_output, "outputs", None):
        return {}
    output = vllm_output.outputs[0]
    logprobs = getattr(output, "logprobs", None)
    if logprobs is None:
        return {}
    step_logprobs = logprobs[0] if isinstance(logprobs, list) else logprobs
    if not isinstance(step_logprobs, dict):
        return {}
    out: Dict[int, float] = {}
    for token_id, value in step_logprobs.items():
        try:
            out[int(token_id)] = float(getattr(value, "logprob", value))
        except Exception:
            continue
    return out


def graph_next_token_logprobs(
    *,
    graph_entity_scores: np.ndarray,
    trie: Any,
    prefix: Sequence[Any],
    allowed_tokens: Sequence[Any],
) -> Dict[Any, float]:
    raw: Dict[Any, float] = {}
    for token in allowed_tokens:
        child_entities = trie.entities_for_prefix(list(prefix) + [token])
        if not child_entities:
            raw[token] = -float("inf")
            continue
        raw[token] = max(float(graph_entity_scores[int(eid)]) for eid in child_entities)
    # Graph scores are prefix representatives first, then converted to a
    # distribution over the same allowed next-token set used by the LLM.
    return log_softmax_dict(raw)


def mean_topk_graph_scores(
    graph_entity_scores: np.ndarray,
    entity_ids: Sequence[int],
    topk: int,
) -> float:
    if not entity_ids:
        return -float("inf")
    values = sorted((float(graph_entity_scores[int(eid)]) for eid in entity_ids), reverse=True)
    k = min(max(int(topk), 1), len(values))
    return float(sum(values[:k]) / k)


def enumerate_token_blocks(
    trie: Any,
    prefix: Sequence[Any],
    block_size: int,
) -> List[Tuple[Any, ...]]:
    blocks: List[Tuple[Any, ...]] = []

    def visit(current_prefix: List[Any], current_block: List[Any]) -> None:
        if len(current_block) == block_size:
            blocks.append(tuple(current_block))
            return
        for token in trie.allowed_tokens(current_prefix):
            visit(current_prefix + [token], current_block + [token])

    visit(list(prefix), [])
    return blocks


def graph_next_block_logprobs(
    *,
    graph_entity_logp: np.ndarray,
    trie: Any,
    prefix: Sequence[Any],
    block_size: int,
) -> Dict[Tuple[Any, ...], float]:
    blocks = enumerate_token_blocks(trie, prefix, block_size)
    parent_entities = trie.entities_for_prefix(prefix)
    parent_logz = logsumexp([float(graph_entity_logp[e]) for e in parent_entities])
    if math.isinf(parent_logz):
        uniform = -math.log(max(len(blocks), 1))
        return {block: uniform for block in blocks}

    raw: Dict[Tuple[Any, ...], float] = {}
    for block in blocks:
        child_entities = trie.entities_for_prefix(list(prefix) + list(block))
        child_logz = logsumexp([float(graph_entity_logp[e]) for e in child_entities])
        raw[block] = child_logz - parent_logz
    return log_softmax_dict(raw)


def graph_next_block_scores(
    *,
    graph_entity_scores: np.ndarray,
    trie: Any,
    prefix: Sequence[Any],
    block_size: int,
    agg: str,
    topk: int,
) -> Dict[Tuple[Any, ...], float]:
    blocks = enumerate_token_blocks(trie, prefix, block_size)
    if agg == "mass":
        parent_entities = trie.entities_for_prefix(prefix)
        parent_logz = logsumexp([float(graph_entity_scores[e]) for e in parent_entities])
        if math.isinf(parent_logz):
            uniform = -math.log(max(len(blocks), 1))
            return {block: uniform for block in blocks}

        raw: Dict[Tuple[Any, ...], float] = {}
        for block in blocks:
            child_entities = trie.entities_for_prefix(list(prefix) + list(block))
            child_logz = logsumexp([float(graph_entity_scores[e]) for e in child_entities])
            raw[block] = child_logz - parent_logz
        return raw

    if agg == "topk_mean_score":
        return {
            block: mean_topk_graph_scores(
                graph_entity_scores,
                trie.entities_for_prefix(list(prefix) + list(block)),
                topk,
            )
            for block in blocks
        }

    raise ValueError(f"Unknown graph prefix aggregation: {agg}")


def maybe_renormalize_llm_logprobs(scores: Dict[int, float], allowed_ids: Sequence[int]) -> Dict[int, float]:
    allowed = set(int(x) for x in allowed_ids)
    raw = {int(tok): float(score) for tok, score in scores.items() if int(tok) in allowed}
    return log_softmax_dict(raw)


def code_item_to_id(item: Any, token_to_id: Dict[str, int]) -> int:
    if isinstance(item, (int, np.integer)):
        return int(item)
    return int(token_to_id[str(item)])


def code_item_to_text(item: Any, token_id_to_token: Dict[int, str]) -> str:
    if isinstance(item, (int, np.integer)):
        return token_id_to_token.get(int(item), str(int(item)))
    return str(item)


def code_items_to_text(items: Sequence[Any], token_id_to_token: Dict[int, str]) -> str:
    return " ".join(code_item_to_text(item, token_id_to_token) for item in items)


def mix_token_logprob(
    llm_logp: float,
    graph_logp: float,
    alpha_graph: float,
    *,
    mix_space: str,
) -> float:
    alpha_graph = min(max(float(alpha_graph), 0.0), 1.0)
    if mix_space != "prob":
        raise ValueError(f"Unknown mix_space={mix_space}")
    if alpha_graph <= 0.0:
        return llm_logp
    if alpha_graph >= 1.0:
        return graph_logp
    left = math.log(max(1.0 - alpha_graph, 1e-12)) + llm_logp
    right = math.log(max(alpha_graph, 1e-12)) + graph_logp
    return logsumexp([left, right])


def mix_normalized_probs(llm_prob: float, graph_prob: float, alpha_graph: float) -> float:
    alpha_graph = min(max(float(alpha_graph), 0.0), 1.0)
    prob = (1.0 - alpha_graph) * float(llm_prob) + alpha_graph * float(graph_prob)
    return math.log(max(prob, 1e-300))


def beam_score_for_mode(beam: Beam, mode: str) -> float:
    if mode == "hybrid":
        return float(beam.score)
    if mode == "llm":
        return float(beam.llm_score)
    if mode == "graph":
        return float(beam.graph_score)
    raise ValueError(f"Unknown score mode: {mode}")


def alpha_for_level(args: argparse.Namespace, level_index: int) -> float:
    if args.alpha_graph_levels:
        expected = int(args.num_code_tokens)
        if len(args.alpha_graph_levels) != expected:
            raise ValueError(
                f"--alpha-graph-levels must contain exactly {expected} values "
                f"when --num-code-tokens={args.num_code_tokens}, "
                f"got {len(args.alpha_graph_levels)}."
            )
        return float(args.alpha_graph_levels[level_index])
    if level_index not in (2, 3):
        return 0.0
    return float(args.final_alpha_graph)


def alpha_for_block(args: argparse.Namespace, block_index: int) -> float:
    if args.alpha_graph_levels:
        expected = int(math.ceil(float(args.num_code_tokens) / float(args.fusion_window)))
        if len(args.alpha_graph_levels) != expected:
            raise ValueError(
                f"--alpha-graph-levels must contain exactly {expected} values "
                f"when --num-code-tokens={args.num_code_tokens} and --fusion-window={args.fusion_window}, "
                f"got {len(args.alpha_graph_levels)}."
            )
        return float(args.alpha_graph_levels[block_index])
    return float(args.alpha_graph)


def build_llm(
    *,
    model_path: str,
    tokenizer_path: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    enable_prefix_caching: bool,
    max_logprobs: int,
    lora_path: Optional[str],
) -> tuple[LLM, Any]:
    llm_kwargs = dict(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_logprobs=max_logprobs,
        enable_prefix_caching=enable_prefix_caching,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
    try:
        import torch

        if torch.cuda.is_available():
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    except Exception:
        pass

    llm = LLM(**llm_kwargs)
    lora_request = None
    if lora_path:
        if LoRARequest is None:
            raise RuntimeError("This vLLM environment does not expose LoRARequest.")
        lora_request = LoRARequest("gter_scode_lora", 1, lora_path)
    return llm, lora_request


def load_generation_tokenizer(tokenizer_path: str, trust_remote_code: bool) -> Any:
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)


def render_prompt_for_generation(tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    prompt = prompt.rstrip() + args.answer_prefix
    if not args.use_chat_template:
        return prompt

    messages = []
    if args.system_message:
        messages.append({"role": "system", "content": args.system_message})
    messages.append({"role": "user", "content": prompt})
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.no_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def prefix_decode_one(
    *,
    llm: LLM,
    lora_request: Any,
    tokenizer: Any,
    prompt: str,
    graph_entity_logp: np.ndarray,
    trie: Any,
    token_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> List[Beam]:
    return prefix_decode_one_multi(
        llm=llm,
        lora_request=lora_request,
        tokenizer=tokenizer,
        prompt=prompt,
        graph_entity_logp=graph_entity_logp,
        trie=trie,
        token_to_id=token_to_id,
        args=args,
    )["hybrid"]


def prefix_decode_one_multi(
    *,
    llm: LLM,
    lora_request: Any,
    tokenizer: Any,
    prompt: str,
    graph_entity_logp: np.ndarray,
    trie: Any,
    token_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, List[Beam]]:
    prompt_text = render_prompt_for_generation(tokenizer, prompt, args)
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    initial_beam = Beam(token_ids=prompt_token_ids[:], code_tokens=[], score=0.0, llm_score=0.0, graph_score=0.0)
    beams_by_mode: Dict[str, List[Beam]] = {mode: [initial_beam] for mode in SCORE_MODES}

    for level_index in range(args.num_code_tokens):
        unique_parent_beams: Dict[Tuple[Any, ...], Beam] = {}
        for beams in beams_by_mode.values():
            for beam in beams:
                key = tuple(beam.code_tokens)
                unique_parent_beams.setdefault(key, beam)

        batched_prompts: List[Dict[str, List[int]]] = []
        batched_params: List[SamplingParams] = []
        batched_beams: List[Beam] = []
        batched_allowed_tokens: List[List[Any]] = []
        batched_allowed_ids: List[List[int]] = []

        for beam in unique_parent_beams.values():
            allowed_tokens = trie.allowed_tokens(beam.code_tokens)
            if not allowed_tokens:
                continue
            allowed_ids = [code_item_to_id(token, token_to_id) for token in allowed_tokens]
            batched_prompts.append({"prompt_token_ids": beam.token_ids})
            batched_params.append(
                SamplingParams(
                    n=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=1,
                    logprobs=min(len(allowed_ids), args.topk_per_step),
                    detokenize=False,
                    skip_special_tokens=False,
                    allowed_token_ids=allowed_ids,
                )
            )
            batched_beams.append(beam)
            batched_allowed_tokens.append(allowed_tokens)
            batched_allowed_ids.append(allowed_ids)

        if not batched_prompts:
            break

        outputs = llm.generate(
            batched_prompts,
            sampling_params=batched_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

        extensions_by_parent_key: Dict[
            Tuple[Any, ...],
            List[Tuple[Any, int, float, float, float]],
        ] = {}
        for beam, allowed_tokens, allowed_ids, output in zip(
            batched_beams, batched_allowed_tokens, batched_allowed_ids, outputs
        ):
            llm_raw = extract_step_topk_logprobs(output)
            llm_logp_by_id = maybe_renormalize_llm_logprobs(llm_raw, allowed_ids)
            graph_logp_by_token = graph_next_token_logprobs(
                graph_entity_scores=graph_entity_logp,
                trie=trie,
                prefix=beam.code_tokens,
                allowed_tokens=allowed_tokens,
            )
            for token in allowed_tokens:
                token_id = code_item_to_id(token, token_to_id)
                llm_logp = llm_logp_by_id.get(token_id, -float("inf"))
                graph_logp = graph_logp_by_token.get(token, -float("inf"))
                if math.isinf(llm_logp) and math.isinf(graph_logp):
                    continue
                mixed_logp = mix_token_logprob(
                    llm_logp,
                    graph_logp,
                    alpha_for_level(args, level_index),
                    mix_space=args.mix_space,
                )
                parent_key = tuple(beam.code_tokens)
                extensions_by_parent_key.setdefault(parent_key, []).append(
                    (token, token_id, llm_logp, graph_logp, mixed_logp)
                )

        next_beams_by_mode: Dict[str, List[Beam]] = {}
        for mode in SCORE_MODES:
            mode_candidates: List[Beam] = []
            for parent_beam in beams_by_mode[mode]:
                parent_key = tuple(parent_beam.code_tokens)
                for token, token_id, llm_logp, graph_logp, mixed_logp in extensions_by_parent_key.get(
                    parent_key, []
                ):
                    mode_candidates.append(
                        Beam(
                            token_ids=parent_beam.token_ids + [token_id],
                            code_tokens=parent_beam.code_tokens + [token],
                            score=parent_beam.score + mixed_logp,
                            llm_score=parent_beam.llm_score
                            + (llm_logp if not math.isinf(llm_logp) else -1e9),
                            graph_score=parent_beam.graph_score
                            + (graph_logp if not math.isinf(graph_logp) else -1e9),
                        )
                    )

            best: Dict[Tuple[Any, ...], Beam] = {}
            for beam in mode_candidates:
                key = tuple(beam.code_tokens)
                if key not in best or beam_score_for_mode(beam, mode) > beam_score_for_mode(best[key], mode):
                    best[key] = beam
            next_beams_by_mode[mode] = sorted(
                best.values(),
                key=lambda item: beam_score_for_mode(item, mode),
                reverse=True,
            )[: args.num_beams]

        beams_by_mode = next_beams_by_mode
        if not any(beams_by_mode.values()):
            break

    return {
        mode: sorted(beams, key=lambda item: beam_score_for_mode(item, mode), reverse=True)
        for mode, beams in beams_by_mode.items()
    }


def append_block_candidates_with_inspect_norm(
    *,
    out: List[Beam],
    parent_beam: Beam,
    block_rows: Sequence[Dict[str, Any]],
    alpha_graph: float,
    args: argparse.Namespace,
) -> None:
    llm_scores = {idx: float(row["llm_logp"]) for idx, row in enumerate(block_rows)}
    graph_scores = {idx: float(row["graph_logp"]) for idx, row in enumerate(block_rows)}
    llm_probs = normalize_prob_dict(llm_scores, args.block_llm_norm)
    graph_probs = normalize_prob_dict(graph_scores, args.block_graph_norm)

    for idx, row in enumerate(block_rows):
        if idx not in llm_probs and idx not in graph_probs:
            continue
        mixed_logp = mix_normalized_probs(
            llm_probs.get(idx, 0.0),
            graph_probs.get(idx, 0.0),
            alpha_graph,
        )
        out.append(
            Beam(
                token_ids=parent_beam.token_ids + list(row["token_ids"]),
                code_tokens=parent_beam.code_tokens + list(row["tokens"]),
                score=parent_beam.score + mixed_logp,
                llm_score=parent_beam.llm_score + float(row["llm_logp"]),
                graph_score=parent_beam.graph_score
                + (float(row["graph_logp"]) if not math.isinf(float(row["graph_logp"])) else -1e9),
            )
        )


def prune_partial_block_states(states: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    limit = int(args.block_inner_beams) if int(args.block_inner_beams) > 0 else int(args.num_beams)
    if limit <= 0:
        return list(states)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for state in states:
        grouped.setdefault(int(state["group_id"]), []).append(state)

    pruned: List[Dict[str, Any]] = []
    for group_states in grouped.values():
        group_states.sort(key=lambda item: float(item["llm_logp"]), reverse=True)
        pruned.extend(group_states[:limit])
    return pruned


def _prefix_decode_one_blockwise_legacy(
    *,
    llm: LLM,
    lora_request: Any,
    tokenizer: Any,
    prompt: str,
    graph_entity_logp: np.ndarray,
    trie: Any,
    token_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> List[Beam]:
    prompt_text = render_prompt_for_generation(tokenizer, prompt, args)
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    beams = [Beam(token_ids=prompt_token_ids[:], code_tokens=[], score=0.0, llm_score=0.0, graph_score=0.0)]
    fusion_window = int(args.fusion_window)

    for block_index, level_start in enumerate(range(0, args.num_code_tokens, fusion_window)):
        block_size = min(fusion_window, int(args.num_code_tokens) - level_start)
        first_prompts: List[Dict[str, List[int]]] = []
        first_params: List[SamplingParams] = []
        first_beams: List[Beam] = []
        first_allowed_tokens: List[List[Any]] = []
        first_allowed_ids: List[List[int]] = []
        first_graph_blocks: List[Dict[Tuple[Any, ...], float]] = []

        for beam in beams:
            allowed_tokens = trie.allowed_tokens(beam.code_tokens)
            if not allowed_tokens:
                continue
            allowed_ids = [code_item_to_id(token, token_to_id) for token in allowed_tokens]
            first_prompts.append({"prompt_token_ids": beam.token_ids})
            first_params.append(
                SamplingParams(
                    n=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=1,
                    logprobs=min(len(allowed_ids), args.topk_per_step),
                    detokenize=False,
                    skip_special_tokens=False,
                    allowed_token_ids=allowed_ids,
                )
            )
            first_beams.append(beam)
            first_allowed_tokens.append(allowed_tokens)
            first_allowed_ids.append(allowed_ids)
            first_graph_blocks.append(
                graph_next_block_scores(
                    graph_entity_scores=graph_entity_logp,
                    trie=trie,
                    prefix=beam.code_tokens,
                    block_size=block_size,
                    agg=args.graph_prefix_agg,
                    topk=args.graph_prefix_topk,
                )
            )

        if not first_prompts:
            break

        first_outputs = llm.generate(
            first_prompts,
            sampling_params=first_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

        candidates: List[Beam] = []
        grouped_parent_beams: Dict[int, Beam] = {}
        grouped_block_rows: Dict[int, List[Dict[str, Any]]] = {}
        second_prompts: List[Dict[str, List[int]]] = []
        second_params: List[SamplingParams] = []
        second_states: List[Tuple[int, Beam, Any, int, float, Dict[Tuple[Any, ...], float]]] = []
        second_allowed_tokens: List[List[Any]] = []
        second_allowed_ids: List[List[int]] = []

        for group_id, (beam, allowed_tokens, allowed_ids, graph_logp_by_block, output) in enumerate(zip(
            first_beams, first_allowed_tokens, first_allowed_ids, first_graph_blocks, first_outputs
        )):
            grouped_parent_beams[group_id] = beam
            grouped_block_rows[group_id] = []
            llm_raw = extract_step_topk_logprobs(output)
            llm_logp_by_id = maybe_renormalize_llm_logprobs(llm_raw, allowed_ids)
            for first_token in allowed_tokens:
                first_id = code_item_to_id(first_token, token_to_id)
                first_llm_logp = llm_logp_by_id.get(first_id, -float("inf"))
                if math.isinf(first_llm_logp):
                    continue

                if block_size == 1:
                    block = (first_token,)
                    graph_logp = graph_logp_by_block.get(block, -float("inf"))
                    grouped_block_rows[group_id].append(
                        {
                            "tokens": block,
                            "token_ids": (first_id,),
                            "llm_logp": first_llm_logp,
                            "graph_logp": graph_logp,
                        }
                    )
                    continue

                second_allowed = trie.allowed_tokens(beam.code_tokens + [first_token])
                if not second_allowed:
                    continue
                second_ids = [code_item_to_id(token, token_to_id) for token in second_allowed]
                second_prompts.append({"prompt_token_ids": beam.token_ids + [first_id]})
                second_params.append(
                    SamplingParams(
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1,
                        logprobs=min(len(second_ids), args.topk_per_step),
                        detokenize=False,
                        skip_special_tokens=False,
                        allowed_token_ids=second_ids,
                    )
                )
                second_states.append((group_id, beam, first_token, first_id, first_llm_logp, graph_logp_by_block))
                second_allowed_tokens.append(second_allowed)
                second_allowed_ids.append(second_ids)

        if block_size == 1:
            for group_id, rows in grouped_block_rows.items():
                append_block_candidates_with_inspect_norm(
                    out=candidates,
                    parent_beam=grouped_parent_beams[group_id],
                    block_rows=rows,
                    alpha_graph=alpha_for_block(args, block_index),
                    args=args,
                )

        if block_size > 1 and second_prompts:
            second_outputs = llm.generate(
                second_prompts,
                sampling_params=second_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            for state, allowed_tokens, allowed_ids, output in zip(
                second_states, second_allowed_tokens, second_allowed_ids, second_outputs
            ):
                group_id, beam, first_token, first_id, first_llm_logp, graph_logp_by_block = state
                llm_raw = extract_step_topk_logprobs(output)
                llm_logp_by_id = maybe_renormalize_llm_logprobs(llm_raw, allowed_ids)
                for second_token in allowed_tokens:
                    second_id = code_item_to_id(second_token, token_to_id)
                    second_llm_logp = llm_logp_by_id.get(second_id, -float("inf"))
                    if math.isinf(second_llm_logp):
                        continue
                    block = (first_token, second_token)
                    llm_block_logp = first_llm_logp + second_llm_logp
                    graph_logp = graph_logp_by_block.get(block, -float("inf"))
                    grouped_block_rows[group_id].append(
                        {
                            "tokens": block,
                            "token_ids": (first_id, second_id),
                            "llm_logp": llm_block_logp,
                            "graph_logp": graph_logp,
                        }
                    )

            for group_id, rows in grouped_block_rows.items():
                append_block_candidates_with_inspect_norm(
                    out=candidates,
                    parent_beam=grouped_parent_beams[group_id],
                    block_rows=rows,
                    alpha_graph=alpha_for_block(args, block_index),
                    args=args,
                )

        best: Dict[Tuple[Any, ...], Beam] = {}
        for beam in candidates:
            key = tuple(beam.code_tokens)
            if key not in best or beam.score > best[key].score:
                best[key] = beam
        beams = sorted(best.values(), key=lambda item: item.score, reverse=True)[: args.num_beams]
        if not beams:
            break

    return sorted(beams, key=lambda item: item.score, reverse=True)


def prefix_decode_one_blockwise(
    *,
    llm: LLM,
    lora_request: Any,
    tokenizer: Any,
    prompt: str,
    graph_entity_logp: np.ndarray,
    trie: Any,
    token_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> List[Beam]:
    prompt_text = render_prompt_for_generation(tokenizer, prompt, args)
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    beams = [Beam(token_ids=prompt_token_ids[:], code_tokens=[], score=0.0, llm_score=0.0, graph_score=0.0)]
    fusion_window = int(args.fusion_window)

    for block_index, level_start in enumerate(range(0, args.num_code_tokens, fusion_window)):
        block_size = min(fusion_window, int(args.num_code_tokens) - level_start)
        candidates: List[Beam] = []
        grouped_parent_beams: Dict[int, Beam] = {}
        grouped_block_rows: Dict[int, List[Dict[str, Any]]] = {}
        grouped_graph_blocks: Dict[int, Dict[Tuple[Any, ...], float]] = {}
        partial_states: List[Dict[str, Any]] = []

        for group_id, beam in enumerate(beams):
            if not trie.allowed_tokens(beam.code_tokens):
                continue
            grouped_parent_beams[group_id] = beam
            grouped_block_rows[group_id] = []
            grouped_graph_blocks[group_id] = graph_next_block_scores(
                graph_entity_scores=graph_entity_logp,
                trie=trie,
                prefix=beam.code_tokens,
                block_size=block_size,
                agg=args.graph_prefix_agg,
                topk=args.graph_prefix_topk,
            )
            partial_states.append(
                {
                    "group_id": group_id,
                    "parent_beam": beam,
                    "tokens": [],
                    "token_ids": [],
                    "llm_logp": 0.0,
                }
            )

        if not partial_states:
            break

        for step_index in range(block_size):
            batched_prompts: List[Dict[str, List[int]]] = []
            batched_params: List[SamplingParams] = []
            batched_states: List[Dict[str, Any]] = []
            batched_allowed_tokens: List[List[Any]] = []
            batched_allowed_ids: List[List[int]] = []

            for state in partial_states:
                beam = state["parent_beam"]
                current_code = beam.code_tokens + list(state["tokens"])
                allowed_tokens = trie.allowed_tokens(current_code)
                if not allowed_tokens:
                    continue
                allowed_ids = [code_item_to_id(token, token_to_id) for token in allowed_tokens]
                batched_prompts.append({"prompt_token_ids": beam.token_ids + list(state["token_ids"])})
                batched_params.append(
                    SamplingParams(
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1,
                        logprobs=min(len(allowed_ids), args.topk_per_step),
                        detokenize=False,
                        skip_special_tokens=False,
                        allowed_token_ids=allowed_ids,
                    )
                )
                batched_states.append(state)
                batched_allowed_tokens.append(allowed_tokens)
                batched_allowed_ids.append(allowed_ids)

            if not batched_prompts:
                partial_states = []
                break

            outputs = llm.generate(
                batched_prompts,
                sampling_params=batched_params,
                use_tqdm=False,
                lora_request=lora_request,
            )

            next_states: List[Dict[str, Any]] = []
            is_last_step = step_index == block_size - 1
            for state, allowed_tokens, allowed_ids, output in zip(
                batched_states, batched_allowed_tokens, batched_allowed_ids, outputs
            ):
                llm_raw = extract_step_topk_logprobs(output)
                llm_logp_by_id = maybe_renormalize_llm_logprobs(llm_raw, allowed_ids)
                for token in allowed_tokens:
                    token_id = code_item_to_id(token, token_to_id)
                    token_llm_logp = llm_logp_by_id.get(token_id, -float("inf"))
                    if math.isinf(token_llm_logp):
                        continue

                    tokens = list(state["tokens"]) + [token]
                    token_ids = list(state["token_ids"]) + [token_id]
                    llm_block_logp = float(state["llm_logp"]) + token_llm_logp
                    group_id = int(state["group_id"])

                    if is_last_step:
                        block = tuple(tokens)
                        graph_logp = grouped_graph_blocks[group_id].get(block, -float("inf"))
                        grouped_block_rows[group_id].append(
                            {
                                "tokens": block,
                                "token_ids": tuple(token_ids),
                                "llm_logp": llm_block_logp,
                                "graph_logp": graph_logp,
                            }
                        )
                    else:
                        next_states.append(
                            {
                                "group_id": group_id,
                                "parent_beam": state["parent_beam"],
                                "tokens": tokens,
                                "token_ids": token_ids,
                                "llm_logp": llm_block_logp,
                            }
                        )

            partial_states = prune_partial_block_states(next_states, args)

        for group_id, rows in grouped_block_rows.items():
            append_block_candidates_with_inspect_norm(
                out=candidates,
                parent_beam=grouped_parent_beams[group_id],
                block_rows=rows,
                alpha_graph=alpha_for_block(args, block_index),
                args=args,
            )

        best: Dict[Tuple[Any, ...], Beam] = {}
        for beam in candidates:
            key = tuple(beam.code_tokens)
            if key not in best or beam.score > best[key].score:
                best[key] = beam
        beams = sorted(best.values(), key=lambda item: item.score, reverse=True)[: args.num_beams]
        if not beams:
            break

    return sorted(beams, key=lambda item: item.score, reverse=True)


def rank_entities_from_code_beams(
    *,
    beams: Sequence[Beam],
    trie: Any,
    graph_entity_logp: np.ndarray,
    id2entity: Dict[int, str],
    token_id_to_token: Dict[int, str],
    topk: int,
    score_mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for beam in beams:
        entities = trie.entities_for_code(beam.code_tokens)
        if score_mode == "llm":
            entities = sorted(entities, key=lambda eid: int(eid))
        else:
            entities = sorted(entities, key=lambda eid: float(graph_entity_logp[eid]), reverse=True)
        for rank_in_code, entity_id in enumerate(entities):
            rank_score = beam_score_for_mode(beam, score_mode)
            candidates.append(
                {
                    "entity_id": int(entity_id),
                    "entity": id2entity.get(int(entity_id), str(entity_id)),
                    "code": code_items_to_text(beam.code_tokens, token_id_to_token),
                    "code_tokens": [code_item_to_text(token, token_id_to_token) for token in beam.code_tokens],
                    "score": float(beam.score),
                    "llm_score": float(beam.llm_score),
                    "graph_prefix_score": float(beam.graph_score),
                    "rank_score": float(rank_score),
                    "rank_score_mode": score_mode,
                    "entity_graph_score": float(graph_entity_logp[int(entity_id)]),
                    "rank_in_code_by_graph": rank_in_code + 1 if score_mode != "llm" else None,
                    "rank_in_code": rank_in_code + 1,
                }
            )
    if score_mode == "llm":
        candidates.sort(key=lambda x: (x["rank_score"], -int(x["rank_in_code"])), reverse=True)
    else:
        candidates.sort(key=lambda x: (x["rank_score"], x["entity_graph_score"]), reverse=True)
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        if item["entity_id"] in seen:
            continue
        seen.add(item["entity_id"])
        dedup.append(item)
        if len(dedup) >= topk:
            break
    return dedup


def code_rows_from_beams(
    beams: Sequence[Beam],
    *,
    token_id_to_token: Dict[int, str],
    topk: int,
    score_mode: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "code": code_items_to_text(beam.code_tokens, token_id_to_token),
            "code_tokens": [code_item_to_text(token, token_id_to_token) for token in beam.code_tokens],
            "score": float(beam.score),
            "llm_score": float(beam.llm_score),
            "graph_prefix_score": float(beam.graph_score),
            "rank_score": float(beam_score_for_mode(beam, score_mode)),
            "rank_score_mode": score_mode,
        }
        for beam in beams[:topk]
    ]


def sample_prompt(sample: Dict[str, Any], prompt_field: str) -> str:
    if prompt_field in sample:
        return str(sample[prompt_field])
    for key in ("prompt", "input"):
        if key in sample:
            return str(sample[key])
    raise KeyError(f"Sample has no prompt field {prompt_field!r}, 'prompt', or 'input'.")


def sample_answer(sample: Dict[str, Any], answer_field: str) -> Any:
    if answer_field in sample:
        return sample[answer_field]
    if "answer" in sample:
        return sample["answer"]
    return sample.get("output")


def sample_target_entity(sample: Dict[str, Any]) -> Optional[int]:
    if sample.get("target_entity") is not None:
        return int(sample["target_entity"])
    meta = sample.get("meta")
    if isinstance(meta, dict):
        for key in ("target_entity", "o_id"):
            if meta.get(key) is not None:
                return int(meta[key])
    return None


def normalize_entity_text(value: Any) -> str:
    return " ".join(str(value).replace("_", " ").strip().lower().split())


def sample_filter_entities(
    sample: Dict[str, Any],
    *,
    code_to_entities: Dict[Tuple[str, ...], List[int]],
    entity_name_to_id: Dict[str, int],
    target_entity: Optional[int],
) -> List[int]:
    filter_ids = set()
    for raw_filter in sample.get("filters", []) or []:
        if isinstance(raw_filter, (int, np.integer)):
            filter_ids.add(int(raw_filter))
            continue

        text = str(raw_filter).strip()
        if not text:
            continue
        if text.lstrip("-").isdigit():
            filter_ids.add(int(text))
            continue

        if CODE_TOKEN_RE.search(text):
            code = split_code_tokens(text)
            filter_ids.update(code_to_entities.get(code, []))
            continue

        entity_id = entity_name_to_id.get(normalize_entity_text(text))
        if entity_id is not None:
            filter_ids.add(int(entity_id))

    if target_entity is not None:
        filter_ids.discard(int(target_entity))
    return sorted(filter_ids)


def sample_query(sample: Dict[str, Any]) -> Any:
    if sample.get("query") is not None:
        return sample.get("query")
    meta = sample.get("meta")
    if isinstance(meta, dict):
        keys = ("s_id", "r_id", "o_id", "t")
        if all(key in meta for key in keys):
            return [meta["s_id"], meta["r_id"], meta["o_id"], meta["t"]]
    return None


def sample_mode(sample: Dict[str, Any]) -> Any:
    if sample.get("mode") is not None:
        return sample.get("mode")
    meta = sample.get("meta")
    if isinstance(meta, dict):
        return meta.get("direction") or meta.get("mode")
    return None


def rank_entity_predictions(
    prediction_entities: Sequence[Dict[str, Any]],
    target: int,
    filters: Optional[Iterable[int]] = None,
) -> float:
    filter_set = {int(entity_id) for entity_id in (filters or []) if int(entity_id) != int(target)}
    rank = 0
    seen = set()
    for pred in prediction_entities:
        entity_id = int(pred["entity_id"])
        if entity_id in seen:
            continue
        seen.add(entity_id)
        if entity_id in filter_set:
            continue
        rank += 1
        if entity_id == int(target):
            return float(rank)
    return float("inf")


def summarize_ranks(ranks: Sequence[float], hits: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        out = {"mrr": 0.0, "num_eval": 0}
        for k in hits:
            out[f"hits@{k}"] = 0.0
        return out
    out = {
        "mrr": float(sum(0.0 if math.isinf(rank) else 1.0 / float(rank) for rank in ranks) / len(ranks)),
        "num_eval": len(ranks),
    }
    for k in hits:
        out[f"hits@{k}"] = float(sum(1 for rank in ranks if rank <= k) / len(ranks))
    return out


def compute_metrics(records: Sequence[Dict[str, Any]], hits: Sequence[int]) -> Dict[str, float]:
    raw_ranks: List[float] = []
    filtered_ranks: List[float] = []
    for record in records:
        target = record.get("target_entity")
        if target is None:
            continue
        target = int(target)
        preds = record.get("metric_prediction_entities") or record.get("prediction_entities", [])
        raw_ranks.append(rank_entity_predictions(preds, target))
        filtered_ranks.append(rank_entity_predictions(preds, target, record.get("filter_entities", [])))

    filtered = summarize_ranks(filtered_ranks, hits)
    raw = summarize_ranks(raw_ranks, hits)
    out = {"metric_type": "filtered", **filtered}
    for key, value in filtered.items():
        out[f"filtered_{key}"] = value
    for key, value in raw.items():
        out[f"raw_{key}"] = value
    return out


def compute_metrics_for_mode(records: Sequence[Dict[str, Any]], hits: Sequence[int], mode: str) -> Dict[str, float]:
    mode_records: List[Dict[str, Any]] = []
    for record in records:
        by_mode = record.get("metric_prediction_entities_by_mode") or {}
        preds = by_mode.get(mode)
        if preds is None:
            if mode == "hybrid":
                preds = record.get("metric_prediction_entities") or record.get("prediction_entities", [])
            else:
                preds = []
        mode_record = dict(record)
        mode_record["metric_prediction_entities"] = preds
        mode_records.append(mode_record)
    return compute_metrics(mode_records, hits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTER-SCode token-prefix decoding with vLLM and dense graph-score mixing.")
    parser.add_argument("--model-path", type=str, required=True, help="Merged model path or warmed base model path.")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA adapter path for vLLM LoRA serving.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Tokenizer path. Defaults to lora path if set, else model path.")
    parser.add_argument("--prompts-jsonl", type=str, required=True)
    parser.add_argument("--graph-score-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--entity-codes", type=str, default="")
    parser.add_argument(
        "--entity-code-map",
        type=str,
        default="",
        help="JSON mapping entity id to semantic-code text. Preferred over --entity-codes when present.",
    )
    parser.add_argument(
        "--entity-trie-pkl",
        type=str,
        default="",
        help="Prebuilt semantic-code trie pickle keyed by tokenizer token ids.",
    )
    parser.add_argument(
        "--code-token-index-path",
        type=str,
        default="",
        help="TSV produced by rq_kmeans, e.g. code_token_embeddings.tsv. Defaults to sibling of --entity-codes.",
    )
    parser.add_argument("--entity2id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--prompt-field", type=str, default="prompt")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--answer-prefix", type=str, default="\n")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-message", type=str, default="")
    parser.add_argument("--no-thinking", action="store_true")
    parser.set_defaults(no_thinking=True)
    parser.add_argument("--max-samples", type=int, default=-1)

    parser.add_argument("--num-code-tokens", type=int, default=4)
    parser.add_argument(
        "--fusion-window",
        type=int,
        default=1,
        choices=[1],
        help="Token-level semantic-code prefix decoding. Block fusion is disabled for this script.",
    )
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--topk-per-step", type=int, default=128)
    parser.add_argument("--topk-entities", type=int, default=50)
    parser.add_argument(
        "--alpha-graph",
        type=float,
        default=0.5,
        help="Legacy graph weight fallback when --final-alpha-graph is not set.",
    )
    parser.add_argument(
        "--alpha-graph-levels",
        type=float,
        nargs="*",
        default=None,
        help="Per-token graph weights for token-level pruning. Expected count is num_code_tokens.",
    )
    parser.add_argument(
        "--mix-space",
        type=str,
        default="prob",
        choices=["prob"],
        help="LLM and graph scores are softmax-normalized over allowed tokens, then mixed in probability space.",
    )
    parser.add_argument(
        "--final-alpha-graph",
        type=float,
        default=None,
        help="Shared graph weight for 3rd/4th-token pruning and final entity reranking. Defaults to --alpha-graph.",
    )
    parser.add_argument(
        "--final-llm-norm",
        type=str,
        default="minmax_softmax",
        choices=["softmax", "minmax_softmax", "none"],
        help="Normalization for final entity LLM scores before LLM/graph fusion.",
    )
    parser.add_argument(
        "--final-graph-norm",
        type=str,
        default="softmax",
        choices=["softmax", "minmax_softmax", "none"],
        help="Normalization for final entity graph scores before LLM/graph fusion.",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--hits", type=int, nargs="+", default=[1, 3, 10, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_vllm_runtime()
    output_dir = resolve_path(args.output_dir)
    setup_logging(output_dir)
    final_alpha_graph = args.alpha_graph if args.final_alpha_graph is None else args.final_alpha_graph
    args.final_alpha_graph = final_alpha_graph
    logger.info("Starting GTER-SCode Prefix Decoding")
    logger.info(json.dumps(sanitized_args(args), ensure_ascii=False, indent=2))

    prompts_path = resolve_path(args.prompts_jsonl, must_exist=True)
    graph_cache = DenseGraphScoreCache(resolve_path(args.graph_score_dir, must_exist=True), args.split)
    id2entity = load_entity_names(resolve_path(args.entity2id, must_exist=False) if args.entity2id else None)

    tokenizer_path = args.tokenizer_path or args.lora_path or args.model_path
    tokenizer = load_generation_tokenizer(tokenizer_path, trust_remote_code=args.trust_remote_code)

    if args.entity_code_map:
        entity_codes = load_entity_code_map(
            resolve_path(args.entity_code_map, must_exist=True),
            num_levels=args.num_code_tokens,
        )
    elif args.entity_codes:
        entity_codes = load_entity_codes(
            resolve_path(args.entity_codes, must_exist=True),
            code_token_index_path=resolve_path(args.code_token_index_path, must_exist=True) if args.code_token_index_path else None,
            num_levels=args.num_code_tokens,
        )
    else:
        raise ValueError("Please pass --entity-code-map or --entity-codes.")
    code_to_entities = build_code_to_entities(entity_codes)
    entity_name_to_id = {normalize_entity_text(name): int(entity_id) for entity_id, name in id2entity.items()}

    token_to_id: Dict[str, int] = {}
    entity_code_ids, token_id_to_token = tokenize_entity_codes(entity_codes, tokenizer)
    for code in entity_codes.values():
        for token in code:
            token_to_id[token] = int(tokenizer.convert_tokens_to_ids(token))

    if args.entity_trie_pkl:
        with resolve_path(args.entity_trie_pkl, must_exist=True).open("rb") as handle:
            trie_payload = pickle.load(handle)
        trie = PrebuiltTokenIdTrie(trie_payload, entity_code_ids)
    else:
        trie = SemanticCodeTrie(entity_codes, token_to_id)
    max_allowed = max(len(trie.allowed_tokens([])), args.topk_per_step, args.num_beams)
    llm, lora_request = build_llm(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=args.enable_prefix_caching,
        max_logprobs=max_allowed,
        lora_path=args.lora_path,
    )

    samples = read_samples(prompts_path)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]

    predictions: List[Dict[str, Any]] = []
    pred_path = output_dir / "predictions_prefix_decode.json"
    metrics_path = output_dir / "metrics.json"
    metrics_by_mode_path = output_dir / "metrics_by_mode.json"

    for index, sample in enumerate(tqdm(samples, desc="prefix decode")):
        query_id = str(sample["id"])
        target_entity = sample_target_entity(sample)
        filter_entities = sample_filter_entities(
            sample,
            code_to_entities=code_to_entities,
            entity_name_to_id=entity_name_to_id,
            target_entity=target_entity,
        )
        graph_row = graph_cache.get(query_id)
        beams_by_mode = prefix_decode_one_multi(
            llm=llm,
            lora_request=lora_request,
            tokenizer=tokenizer,
            prompt=sample_prompt(sample, args.prompt_field),
            graph_entity_logp=graph_row,
            trie=trie,
            token_to_id=token_to_id,
            args=args,
        )

        code_rows_by_mode = {
            mode: code_rows_from_beams(
                beams,
                token_id_to_token=token_id_to_token,
                topk=args.topk_entities,
                score_mode=mode,
            )
            for mode, beams in beams_by_mode.items()
        }
        metric_topk = max(args.topk_entities, max(args.hits, default=args.topk_entities) + len(filter_entities))
        metric_entity_rows_by_mode = {
            mode: rank_entities_from_code_beams(
                beams=beams,
                trie=trie,
                graph_entity_logp=graph_row,
                id2entity=id2entity,
                token_id_to_token=token_id_to_token,
                topk=metric_topk,
                score_mode=mode,
            )
            for mode, beams in beams_by_mode.items()
        }
        entity_rows_by_mode = {
            mode: rows[: args.topk_entities]
            for mode, rows in metric_entity_rows_by_mode.items()
        }
        code_rows = code_rows_by_mode["hybrid"]
        entity_rows = entity_rows_by_mode["hybrid"]
        metric_entity_rows = metric_entity_rows_by_mode["hybrid"]
        predictions.append(
            {
                "id": query_id,
                "query": sample_query(sample),
                "mode": sample_mode(sample),
                "target_entity": target_entity,
                "answer": sample_answer(sample, args.answer_field),
                "filters": sample.get("filters", []),
                "filter_entities": filter_entities,
                "prediction_codes": code_rows,
                "prediction_entities": entity_rows,
                "metric_prediction_entities": metric_entity_rows,
                "prediction_codes_llm": code_rows_by_mode["llm"],
                "prediction_entities_llm": entity_rows_by_mode["llm"],
                "metric_prediction_entities_llm": metric_entity_rows_by_mode["llm"],
                "prediction_codes_graph": code_rows_by_mode["graph"],
                "prediction_entities_graph": entity_rows_by_mode["graph"],
                "metric_prediction_entities_graph": metric_entity_rows_by_mode["graph"],
                "prediction_codes_by_mode": code_rows_by_mode,
                "prediction_entities_by_mode": entity_rows_by_mode,
                "metric_prediction_entities_by_mode": metric_entity_rows_by_mode,
            }
        )
        if args.save_every > 0 and (index + 1) % args.save_every == 0:
            write_json(pred_path, predictions)
            metrics_by_mode = {mode: compute_metrics_for_mode(predictions, args.hits, mode) for mode in SCORE_MODES}
            write_json(metrics_path, metrics_by_mode["hybrid"])
            write_json(metrics_by_mode_path, metrics_by_mode)

    metrics_by_mode = {mode: compute_metrics_for_mode(predictions, args.hits, mode) for mode in SCORE_MODES}
    metrics = metrics_by_mode["hybrid"]
    write_json(pred_path, predictions)
    write_json(metrics_path, metrics)
    write_json(metrics_by_mode_path, metrics_by_mode)
    write_json(
        output_dir / "run_summary.json",
        {
            "num_samples": len(samples),
            "num_predictions": len(predictions),
            "alpha_graph": args.alpha_graph,
            "alpha_graph_levels": args.alpha_graph_levels,
            "final_alpha_graph": final_alpha_graph,
            "final_llm_norm": args.final_llm_norm,
            "final_graph_norm": args.final_graph_norm,
            "fusion_window": args.fusion_window,
            "mix_space": args.mix_space,
            "paths": sanitized_args(args),
            "score_formula": (
                "Token-level prefix decoding only. For each allowed next token c, "
                "graph_score(c)=max graph_entity_score(e) over entities under prefix+c; "
                "p_llm=softmax(llm_score over allowed tokens), "
                "p_graph=softmax(graph_score over allowed tokens), "
                "p_mix_t(c|prefix)=(1-alpha_t)*p_llm(c|prefix)+alpha_t*p_graph(c|prefix), "
                "where alpha_t comes from alpha_graph_levels when provided, otherwise "
                "1st/2nd-token pruning is LLM-only and 3rd/4th-token pruning uses final_alpha_graph. "
                "Beam frontiers are tracked in parallel for hybrid score, pure llm_score, and pure graph_score. "
                "Hybrid and graph entity rankings use entity_graph_score as a tie-breaker inside decoded codes; "
                "LLM entity ranking keeps entities under the same code in entity-id order."
            ),
            "graph_score_type": graph_cache.summary.get("score_type"),
            "metrics": metrics,
            "metrics_by_mode": metrics_by_mode,
        },
    )
    logger.info("Saved predictions.")
    logger.info("Saved mode metrics.")
    logger.info("Metrics: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
