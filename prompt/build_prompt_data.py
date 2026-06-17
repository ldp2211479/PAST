#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Sequence, Set, Tuple

from wise.prompt.rbmh_stage2_sampler import RBMHStage2Sampler, TimedFact

from tqdm import tqdm


INSTRUCTION_TEXT = (
    "Given the historical interactions of a subject entity, "
    "predict the missing entity in the query. output only its semantic code. Do not output the entity name or explanation."
)


class TKGBuilder:
    def __init__(
        self,
        dataset_name: str,
        base_data_dir: str,
        history_len: int = 50,
        sampling_scheme: str = "simple_plus_rbmh_stage2",
        rbmh_gamma1: float = 0.6,
        rbmh_gamma2: float = 0.6,
        rbmh_gamma3: float = 0.01,
        rbmh_gamma4: float = 0.1,
        rbmh_time_delta: float = 24.0,
        rbmh_seed: int = 42,
        rbmh_top_multiplier: int = 10,
        rbmh_max_hop: int = 0,
        rbmh_deterministic_top: bool = False,
        rbmh_candidate_recent_limit: int = 256,
    ):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(base_data_dir, dataset_name)
        self.history_len = history_len
        self.sampling_scheme = sampling_scheme
        self.rbmh_gamma1 = rbmh_gamma1
        self.rbmh_gamma2 = rbmh_gamma2
        self.rbmh_gamma3 = rbmh_gamma3
        self.rbmh_gamma4 = rbmh_gamma4
        self.rbmh_time_delta = rbmh_time_delta
        self.rbmh_seed = rbmh_seed
        self.rbmh_top_multiplier = rbmh_top_multiplier
        self.rbmh_max_hop = rbmh_max_hop
        self.rbmh_deterministic_top = rbmh_deterministic_top
        self.rbmh_candidate_recent_limit = rbmh_candidate_recent_limit

        self.id2entity = self._load_map("entity2id.txt")
        self.id2relation = self._load_map("relation2id.txt")

        self.train_facts = self._load_facts("train.txt")
        self.valid_facts = self._load_facts("valid.txt")
        self.test_facts = self._load_facts("test.txt")
        self.all_facts = self.train_facts + self.valid_facts + self.test_facts

        self.history_index = self._build_history_index()
        self.filters_index = self._build_filters_index()
        self.rbmh_sampler = RBMHStage2Sampler(
            facts=self.all_facts,
            history_len=self.history_len,
            gamma1=self.rbmh_gamma1,
            gamma2=self.rbmh_gamma2,
            gamma3=self.rbmh_gamma3,
            gamma4=self.rbmh_gamma4,
            time_delta=self.rbmh_time_delta,
            seed=self.rbmh_seed,
            top_multiplier=self.rbmh_top_multiplier,
            max_hop=self.rbmh_max_hop,
            deterministic_top=self.rbmh_deterministic_top,
            candidate_recent_limit=self.rbmh_candidate_recent_limit,
        )

    def _load_map(self, filename: str) -> Dict[int, str]:
        path = os.path.join(self.data_dir, filename)
        mapping: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                name, entity_id = parts[:2]
                mapping[int(entity_id)] = name.replace("_", " ")
        return mapping

    def _load_facts(self, filename: str) -> List[Tuple[int, int, int, int]]:
        path = os.path.join(self.data_dir, filename)
        facts: List[Tuple[int, int, int, int]] = []
        if not os.path.exists(path):
            return facts

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                s, r, o, t = map(int, parts[:4])
                facts.append((s, r, o, t))
        return facts

    def _build_history_index(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
        history_index: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
        for s, r, o, t in sorted(self.all_facts, key=lambda x: x[3]):
            history_index[s].append((t, s, r, o))
            history_index[o].append((t, s, r, o))
        return history_index

    def _build_filters_index(self) -> Dict[Tuple[int, int, int, str], Set[int]]:
        filters_index: Dict[Tuple[int, int, int, str], Set[int]] = defaultdict(set)
        for s, r, o, t in self.all_facts:
            filters_index[(s, r, t, "forward")].add(o)
            filters_index[(o, r, t, "backward")].add(s)
        return filters_index

    def _get_simple_history(self, entity_id: int, query_time: int, test_no_history=False) -> List[TimedFact]:
        facts = self.history_index.get(entity_id, [])
        if test_no_history:
            facts = []
        selected: List[TimedFact] = []

        for t, s, r, o in reversed(facts):
            if t >= query_time:
                continue
            selected.append((t, s, r, o))
            if len(selected) >= self.history_len:
                break
        return selected

    def get_history(
        self,
        entity_id: int,
        query_time: int,
        *,
        answer_entity_id: int,
        relation_id: int,
        direction: str,
        sample_global_id: int,
        test_no_history=False,
    ) -> Tuple[str, Set[int]]:
        selected = self._get_simple_history(entity_id, query_time, test_no_history)
        if (
            not test_no_history
            and self.sampling_scheme == "simple_plus_rbmh_stage2"
            and len(selected) < self.history_len
        ):
            selected = self.rbmh_sampler.supplement(
                base_history=selected,
                query_entity=entity_id,
                answer_entity=answer_entity_id,
                relation_id=relation_id,
                query_time=query_time,
                direction=direction,
                sample_global_id=sample_global_id,
            )

        visible_entities: Set[int] = set()
        lines: List[str] = []
        for t, s, r, o in selected:
            visible_entities.add(s)
            visible_entities.add(o)
            s_name = self.id2entity[s]
            r_name = self.id2relation[r]
            o_name = self.id2entity[o]
            lines.append(f"{t}: [{s_name}\t{r_name}\t{o_name}]")

        return "\n".join(lines), visible_entities

    def _build_sample(
        self,
        *,
        global_id: int,
        query_entity_id: int,
        relation_id: int,
        answer_entity_id: int,
        time_id: int,
        split_name: str,
        direction: str,
        test_no_history = False,
    ) -> Dict:
        query_entity_name = self.id2entity[query_entity_id]
        answer_entity_name = self.id2entity[answer_entity_id]
        relation_name = self.id2relation[relation_id if direction == "forward" else relation_id - len(self.id2relation)]

        history_text, visible_entities = self.get_history(
            query_entity_id,
            time_id,
            answer_entity_id=answer_entity_id,
            relation_id=relation_id,
            direction=direction,
            sample_global_id=global_id,
            test_no_history=test_no_history,
        )
        in_history = answer_entity_id in visible_entities

        if direction == "forward":
            query_text = f"{time_id}: [{query_entity_name}\t{relation_name}\t?]"
            filter_ids = self.filters_index[(query_entity_id, relation_id, time_id, "forward")]
        else:
            query_text = f"{time_id}: [?\t{relation_name}\t{query_entity_name}]"
            filter_ids = self.filters_index[(query_entity_id, relation_id - len(self.id2relation), time_id, "backward")]

        filter_names = [self.id2entity[x] for x in filter_ids if x != answer_entity_id]

        return {
            "id": global_id,
            "instruction": "",
            "input": (
                f"### Instruction ###\n{INSTRUCTION_TEXT}\n"
                f"### Subject ###\n{query_entity_name}\n"
                f"### History ###\n{history_text}\n"
                f"### Query ###\n{query_text}\n"
                f"### Answer ###\n"
            ),
            "output": f"{answer_entity_name}",
            "filters": filter_names,
            "meta": {
                "s_id": query_entity_id,
                "r_id": relation_id,
                "o_id": answer_entity_id,
                "t": time_id,
                "split": split_name,
                "direction": direction,
                "in_history": in_history,
            },
        }

    def generate_samples(self, facts: Sequence[Tuple[int, int, int, int]], split_name: str, skip_first_snapshot: bool, test_no_history=False) -> List[Dict]:
        samples: List[Dict] = []
        facts_sorted = sorted(facts, key=lambda x: x[3])
        snapshots = [list(group) for _, group in groupby(facts_sorted, key=lambda x: x[3])]
        start_index = 1 if skip_first_snapshot else 0

        global_id = 0
        for snapshot in tqdm(snapshots[start_index:], desc=f"build {split_name}"):
            for s, r, o, t in snapshot:
                samples.append(
                    self._build_sample(
                        global_id=global_id,
                        query_entity_id=s,
                        relation_id=r,
                        answer_entity_id=o,
                        time_id=t,
                        split_name=split_name,
                        direction="forward",
                        test_no_history=test_no_history,
                    )
                )
                global_id += 1

            for s, r, o, t in snapshot:
                samples.append(
                    self._build_sample(
                        global_id=global_id,
                        query_entity_id=o,
                        relation_id=r + len(self.id2relation),
                        answer_entity_id=s,
                        time_id=t,
                        split_name=split_name,
                        direction="backward",
                        test_no_history=test_no_history,
                    )
                )
                global_id += 1

        return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DGS-Debias prompt data.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--base_data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument(
        "--sampling_scheme",
        type=str,
        default="simple_plus_rbmh_stage2",
        choices=["simple", "simple_plus_rbmh_stage2"],
    )
    parser.add_argument("--rbmh_gamma1", type=float, default=0.6)
    parser.add_argument("--rbmh_gamma2", type=float, default=0.6)
    parser.add_argument("--rbmh_gamma3", type=float, default=0.01)
    parser.add_argument("--rbmh_gamma4", type=float, default=0.1)
    parser.add_argument("--rbmh_time_delta", type=float, default=24.0)
    parser.add_argument("--rbmh_seed", type=int, default=42)
    parser.add_argument("--rbmh_top_multiplier", type=int, default=10)
    parser.add_argument("--rbmh_max_hop", type=int, default=0)
    parser.add_argument("--rbmh_deterministic_top", action="store_true")
    parser.add_argument("--rbmh_candidate_recent_limit", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = TKGBuilder(
        dataset_name=args.dataset_name,
        base_data_dir=args.base_data_dir,
        history_len=args.history_len,
        sampling_scheme=args.sampling_scheme,
        rbmh_gamma1=args.rbmh_gamma1,
        rbmh_gamma2=args.rbmh_gamma2,
        rbmh_gamma3=args.rbmh_gamma3,
        rbmh_gamma4=args.rbmh_gamma4,
        rbmh_time_delta=args.rbmh_time_delta,
        rbmh_seed=args.rbmh_seed,
        rbmh_top_multiplier=args.rbmh_top_multiplier,
        rbmh_max_hop=args.rbmh_max_hop,
        rbmh_deterministic_top=args.rbmh_deterministic_top,
        rbmh_candidate_recent_limit=args.rbmh_candidate_recent_limit,
    )

    save_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    train_samples = builder.generate_samples(builder.train_facts, "train", skip_first_snapshot=True)
    valid_samples = builder.generate_samples(builder.valid_facts, "valid", skip_first_snapshot=False)
    test_samples = builder.generate_samples(builder.test_facts, "test", skip_first_snapshot=False)
    test_samples_no_his = builder.generate_samples(builder.test_facts, "test no history", skip_first_snapshot=False, test_no_history=True)

    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "valid.json"), "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "test_no_his.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples_no_his, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
