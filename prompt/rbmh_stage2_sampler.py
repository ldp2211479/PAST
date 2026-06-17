from __future__ import annotations

import bisect
import heapq
import math
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Set, Tuple

Fact = Tuple[int, int, int, int]       # (s, r, o, t)
TimedFact = Tuple[int, int, int, int]  # (t, s, r, o)


class RBMHStage2Sampler:
    """Local-fast RBMH-Stage-2-style supplement sampler.

    This sampler matches the latest Scheme B in retrieval_recall_eval-local-fast:

    1. The caller first builds the original simple incident history.
    2. If that history is shorter than ``history_len``, this class expands from
       entities already visible in the simple history.
    3. Candidate facts are scored with the RBMH Stage-2 weight
       ``w = w_n * w_f * (w_t + w_c + w_cp)``.
    4. The sampler keeps the top ``top_multiplier * M`` candidates and samples
       ``M`` supplemental facts with the local-fast weighted sampler.

    This is intentionally not the original full-scan RBMH variant. It follows
    the local expansion variant used by the provided recall-evaluation code.
    """

    def __init__(
        self,
        facts: Sequence[Fact],
        history_len: int = 50,
        gamma1: float = 0.6,
        gamma2: float = 0.6,
        gamma3: float = 0.01,
        gamma4: float = 0.1,
        time_delta: float = 24.0,
        seed: int = 42,
        top_multiplier: int = 10,
        max_hop: int = 0,
        deterministic_top: bool = False,
        candidate_recent_limit: int = 256,
    ) -> None:
        self.facts = list(facts)
        self.history_len = int(history_len)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.gamma3 = float(gamma3)
        self.gamma4 = float(gamma4)
        self.time_delta = max(float(time_delta), 1e-12)
        self.seed = int(seed)
        self.top_multiplier = max(1, int(top_multiplier))
        self.max_hop = max_hop if max_hop and max_hop > 0 else None
        self.deterministic_top = bool(deterministic_top)
        self.candidate_recent_limit = max(0, int(candidate_recent_limit))

        self.triple_times = self._build_triple_times()
        self.pair_times = self._build_pair_times()
        self.entity_history, self.entity_history_times = self._build_entity_history()

    def _build_entity_history(self) -> Tuple[Dict[int, List[TimedFact]], Dict[int, List[int]]]:
        entity_history: DefaultDict[int, List[TimedFact]] = defaultdict(list)
        for s, r, o, t in sorted(self.facts, key=lambda x: x[3]):
            fact = (t, s, r, o)
            entity_history[s].append(fact)
            entity_history[o].append(fact)
        history = dict(entity_history)
        times = {entity_id: [fact[0] for fact in facts] for entity_id, facts in history.items()}
        return history, times

    def _build_triple_times(self) -> Dict[Tuple[int, int, int], List[int]]:
        triple_times: DefaultDict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for s, r, o, t in self.facts:
            triple_times[(s, r, o)].append(t)
        for times in triple_times.values():
            times.sort()
        return dict(triple_times)

    def _build_pair_times(self) -> Dict[Tuple[int, int], List[int]]:
        pair_times: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        for s, _, o, t in self.facts:
            key = (s, o) if s <= o else (o, s)
            pair_times[key].append(t)
        for times in pair_times.values():
            times.sort()
        return dict(pair_times)

    @staticmethod
    def _entities_in_history(history: Sequence[TimedFact]) -> Set[int]:
        visible: Set[int] = set()
        for _, s, _, o in history:
            visible.add(s)
            visible.add(o)
        return visible

    def _triple_count_before(self, s: int, r: int, o: int, query_time: int) -> int:
        times = self.triple_times.get((s, r, o), [])
        return bisect.bisect_left(times, query_time)

    def _pair_count_before(self, s: int, o: int, query_time: int) -> int:
        key = (s, o) if s <= o else (o, s)
        times = self.pair_times.get(key, [])
        return bisect.bisect_left(times, query_time)

    def _stage2_weight(
        self,
        *,
        cand_t: int,
        cand_s: int,
        cand_r: int,
        cand_o: int,
        query_entity: int,
        query_time: int,
        hop_dict: Dict[int, int],
        base_context_entities: Set[int],
    ) -> float:
        hop_s = hop_dict.get(cand_s)
        hop_o = hop_dict.get(cand_o)
        if hop_s is None or hop_o is None:
            return 0.0

        w_n = math.exp(-self.gamma1 * (hop_s + hop_o - 1))

        n_spo = max(1, self._triple_count_before(cand_s, cand_r, cand_o, query_time))
        w_f = 1.0 / (self.gamma2 * math.log(n_spo) + 1.0)

        w_t = math.exp(-self.gamma3 * ((query_time - cand_t) / self.time_delta))

        n_so = self._pair_count_before(cand_s, cand_o, query_time)
        log_term = math.log(1.0 + self.gamma4 * n_so)
        w_c = log_term / (1.0 + log_term)

        w_cp = 1.0 if (cand_s in base_context_entities or cand_o in base_context_entities) else 0.0

        weight = w_n * w_f * (w_t + w_c + w_cp)
        if not math.isfinite(weight) or weight <= 0.0:
            return 0.0
        return weight

    def _weighted_sample_without_replacement_local_fast(
        self,
        scored: List[Tuple[float, TimedFact]],
        sample_size: int,
        rng: random.Random,
    ) -> List[TimedFact]:
        if sample_size <= 0 or not scored:
            return []
        scored = [(w, f) for w, f in scored if w > 0.0]
        if not scored:
            return []

        top_k = min(len(scored), self.top_multiplier * sample_size)
        pool = heapq.nlargest(top_k, scored, key=lambda x: x[0])
        if self.deterministic_top:
            return [f for _, f in pool[:sample_size]]

        draws = min(sample_size, len(pool))
        keyed: List[Tuple[float, TimedFact]] = []
        for w, fact in pool:
            if w <= 0.0:
                continue
            key = math.log(max(rng.random(), 1e-12)) / w
            keyed.append((key, fact))
        if not keyed:
            return []
        return [fact for _, fact in heapq.nlargest(draws, keyed, key=lambda x: x[0])]

    def _local_scored_candidates(
        self,
        *,
        query_entity: int,
        query_time: int,
        base_context_entities: Set[int],
        base_fact_set: Set[TimedFact],
    ) -> List[Tuple[float, TimedFact]]:
        scored_candidates: List[Tuple[float, TimedFact]] = []
        seen: Set[TimedFact] = set()

        base_hops: Dict[int, int] = {query_entity: 0}
        for entity_id in base_context_entities:
            if entity_id != query_entity:
                base_hops[entity_id] = min(base_hops.get(entity_id, 1), 1)

        for seed_entity in base_context_entities:
            facts = self.entity_history.get(seed_entity)
            if not facts:
                continue
            times = self.entity_history_times[seed_entity]
            end = bisect.bisect_left(times, query_time)
            if end <= 0:
                continue
            start = 0
            if self.candidate_recent_limit > 0:
                start = max(0, end - self.candidate_recent_limit)

            seed_hop = base_hops.get(seed_entity, 1)
            for idx in range(end - 1, start - 1, -1):
                cand_t, cand_s, cand_r, cand_o = facts[idx]
                if cand_s == query_entity:
                    continue
                fact = (cand_t, cand_s, cand_r, cand_o)
                if fact in base_fact_set or fact in seen:
                    continue
                seen.add(fact)

                local_hops = dict(base_hops)
                if cand_s == seed_entity:
                    local_hops[cand_o] = min(local_hops.get(cand_o, seed_hop + 1), seed_hop + 1)
                elif cand_o == seed_entity:
                    local_hops[cand_s] = min(local_hops.get(cand_s, seed_hop + 1), seed_hop + 1)
                else:
                    continue

                if self.max_hop is not None:
                    if local_hops.get(cand_s, self.max_hop + 1) > self.max_hop:
                        continue
                    if local_hops.get(cand_o, self.max_hop + 1) > self.max_hop:
                        continue

                w = self._stage2_weight(
                    cand_t=cand_t,
                    cand_s=cand_s,
                    cand_r=cand_r,
                    cand_o=cand_o,
                    query_entity=query_entity,
                    query_time=query_time,
                    hop_dict=local_hops,
                    base_context_entities=base_context_entities,
                )
                if w > 0.0:
                    scored_candidates.append((w, fact))
        return scored_candidates

    def supplement(
        self,
        *,
        base_history: Sequence[TimedFact],
        query_entity: int,
        answer_entity: int,
        relation_id: int,
        query_time: int,
        direction: str,
        sample_global_id: int,
    ) -> List[TimedFact]:
        base_history = list(base_history)
        if len(base_history) >= self.history_len:
            return base_history[: self.history_len]

        residual = self.history_len - len(base_history)
        if residual <= 0:
            return base_history[: self.history_len]

        base_context_entities = self._entities_in_history(base_history)
        base_fact_set = set(base_history)
        scored_candidates = self._local_scored_candidates(
            query_entity=query_entity,
            query_time=query_time,
            base_context_entities=base_context_entities,
            base_fact_set=base_fact_set,
        )

        direction_offset = 0 if direction == "forward" else 1
        stable_seed = (
            self.seed
            + sample_global_id * 1_000_003
            + query_entity * 9_176
            + answer_entity * 131
            + relation_id * 97
            + query_time * 389
            + direction_offset
        ) & 0xFFFFFFFF
        rng = random.Random(stable_seed)
        supplements = self._weighted_sample_without_replacement_local_fast(scored_candidates, residual, rng)
        return (base_history + supplements)[: self.history_len]
