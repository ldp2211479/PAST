import csv
import json
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append(".")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
from src.numpy_compat import load_pickle_npy
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_snapshot_timestamps(data_array):
    timestamps = []
    latest_t = None
    for row in data_array:
        t = int(row[3])
        if latest_t != t:
            timestamps.append(t)
            latest_t = t
    return timestamps


class DenseGraphScoreWriter:
    def __init__(self, output_dir, split, num_queries, num_entities):
        self.output_dir = output_dir
        self.split = split
        self.num_queries = int(num_queries)
        self.num_entities = int(num_entities)
        os.makedirs(self.output_dir, exist_ok=True)
        self.score_path = os.path.join(self.output_dir, f"{self.split}_graph_scores.npy")
        self.query_path = os.path.join(self.output_dir, f"{self.split}_graph_queries.jsonl")
        self.summary_path = os.path.join(self.output_dir, f"{self.split}_graph_scores_summary.json")
        self.scores = np.lib.format.open_memmap(
            self.score_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_queries, self.num_entities),
        )
        self.query_handle = open(self.query_path, "w", encoding="utf-8")
        self.row_index = 0

    def write(self, metadata, score_row):
        if self.row_index >= self.num_queries:
            raise IndexError(
                f"Too many graph-score rows for split={self.split}: "
                f"row_index={self.row_index}, expected={self.num_queries}"
            )
        if torch.is_tensor(score_row):
            score_row = score_row.detach().float().cpu().numpy()
        score_row = np.asarray(score_row, dtype=np.float32)
        if score_row.shape != (self.num_entities,):
            raise ValueError(f"Graph score row shape mismatch: got={score_row.shape}, expected={(self.num_entities,)}")
        row = dict(metadata)
        row["row_index"] = int(self.row_index)
        self.scores[self.row_index] = score_row
        self.query_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.row_index += 1

    def close(self):
        self.scores.flush()
        self.query_handle.close()
        if self.row_index != self.num_queries:
            raise ValueError(
                f"Written graph-score rows mismatch for split={self.split}: "
                f"wrote={self.row_index}, expected={self.num_queries}"
            )
        summary = {
            "split": self.split,
            "num_queries": self.num_queries,
            "num_entities": self.num_entities,
            "score_type": "log_probability",
            "row_order": "snapshot_order_tail_rows_then_head_rows",
            "score_file": os.path.basename(self.score_path),
            "query_file": os.path.basename(self.query_path),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def count_dense_graph_queries(target_list):
    return int(sum(len(snap) for snap in target_list) * 2)


def update_dict(subg_arr, s_to_sro, sr_to_sro,sro_to_fre, num_rels):
    # 根据输入的每一个时间的图来更新查询查询
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)
        
def e2r(triplets, num_rels):
    # 统计同一个查询实体连接不同的关系
    src, rel, dst = triplets.transpose()
    # get all relations
    # uniq_e = np.concatenate((src, dst))
    uniq_e = np.unique(src)
    # generate r2e
    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)
        # e_to_r[dst].add(rel+num_rels)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    return [uniq_e, r_len, r_idx]

def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,num_nodes, num_rels, use_cuda, gpu):
    q_tri, q_tri_inv = sample_history_cache_arrays(subg_arr, sr_to_sro, triples, num_rels)
    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu) 
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv


def union_nested_values(values):
    merged = set()
    for value in values:
        merged.update(value)
    return merged


def sample_history_cache_arrays(subg_arr, sr_to_sro, triples, num_rels):
    if len(subg_arr) == 0 or len(triples) == 0:
        empty = np.empty((0, 4), dtype=np.int64)
        return empty, empty

    inverse_triples = triples[:, [2, 1, 0]].copy()
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels

    src_set = set(int(x) for x in triples[:, 0])
    dst_set = set(int(x) for x in triples[:, 0])
    er_list = list(set((int(tri[0]), int(tri[1])) for tri in triples))
    er_list_inv = list(set((int(tri[0]), int(tri[1])) for tri in inverse_triples))

    inverse_subg = subg_arr[:, [2, 1, 0]].copy()
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'})

    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic = pd.DataFrame({'sr': keys, 'dst': values})

    dst_df = df_dic[df_dic['sr'].isin(er_list)]
    two_ent = union_nested_values(dst_df['dst'].values) if len(dst_df) else set()
    all_ent = list(src_set | two_ent)
    result = subg_df[subg_df['src'].isin(all_ent)]

    dst_df_inv = df_dic[df_dic['sr'].isin(er_list_inv)]
    two_ent_inv = union_nested_values(dst_df_inv['dst'].values) if len(dst_df_inv) else set()
    all_ent_inv = list(dst_set | two_ent_inv)
    result_inv = subg_df[subg_df['src'].isin(all_ent_inv)]
    return result.to_numpy(dtype=np.int64), result_inv.to_numpy(dtype=np.int64)


def logcl_history_cache_ready(dataset_dir, num_train_snapshots):
    his_dict = os.path.join(dataset_dir, 'his_dict', 'train_s_r.npy')
    if not os.path.exists(his_dict):
        return False
    for train_sample_num in range(1, num_train_snapshots):
        for subdir, prefix in [('his_graph_for', 'train_s_r'), ('his_graph_inv', 'train_o_r')]:
            path = os.path.join(dataset_dir, subdir, '{}_{}.npy'.format(prefix, train_sample_num))
            if not os.path.exists(path):
                return False
    return True


def ensure_logcl_history_cache(dataset, data_root, train_list, num_rels, force_rebuild=False):
    dataset_dir = os.path.join(data_root, dataset)
    if logcl_history_cache_ready(dataset_dir, len(train_list)) and not force_rebuild:
        return

    print("Building LogCL history cache for {} under {}".format(dataset, dataset_dir))
    for subdir in ['his_graph_for', 'his_graph_inv', 'his_dict']:
        os.makedirs(os.path.join(dataset_dir, subdir), exist_ok=True)

    s_to_sro = defaultdict(set)
    sr_to_sro = defaultdict(set)
    for train_sample_num in tqdm(range(len(train_list)), desc="build LogCL cache"):
        if train_sample_num == 0:
            continue
        update_dict(train_list[train_sample_num - 1], s_to_sro, sr_to_sro, None, num_rels)
        subg_arr = np.concatenate(train_list[:train_sample_num])
        triples = train_list[train_sample_num]
        sub_snap, sub_snap_inv = sample_history_cache_arrays(subg_arr, sr_to_sro, triples, num_rels)
        np.save(os.path.join(dataset_dir, 'his_graph_for', 'train_s_r_{}.npy'.format(train_sample_num)), sub_snap)
        np.save(os.path.join(dataset_dir, 'his_graph_inv', 'train_o_r_{}.npy'.format(train_sample_num)), sub_snap_inv)

    np.save(os.path.join(dataset_dir, 'his_dict', 'train_s_r.npy'), sr_to_sro)
    print("Saved LogCL history cache for {}".format(dataset))




def clean_logcl_state_dict(state_dict):
    """Remove dynamically registered RGCN relation-embedding keys.

    LogCL's RGCN layers may attach `rel_emb` during forward. Those dynamic
    keys can appear in saved checkpoints, but they are not present in a newly
    constructed model before forward. We drop only these duplicated dynamic
    keys and keep strict loading for all real parameters.
    """
    cleaned = {}
    dropped = []
    for k, v in state_dict.items():
        if k.endswith(".rel_emb") and ".layers." in k:
            dropped.append(k)
            continue
        cleaned[k] = v
    if dropped:
        print("[LogCL checkpoint] dropped dynamic rel_emb keys:", len(dropped))
        for k in dropped[:20]:
            print("  -", k)
        if len(dropped) > 20:
            print("  ...")
    return cleaned


def _is_dynamic_logcl_rel_emb_key(key):
    return key.endswith(".rel_emb") and ".layers." in key


def load_logcl_state_dict(model, state_dict):
    """Load LogCL checkpoints while ignoring duplicated dynamic rel_emb keys."""
    incompatible = model.load_state_dict(clean_logcl_state_dict(state_dict), strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    real_missing = [k for k in missing_keys if not _is_dynamic_logcl_rel_emb_key(k)]

    if real_missing or unexpected_keys:
        raise RuntimeError(
            "Error(s) in loading LogCL checkpoint:\n"
            "    Missing key(s): {}\n"
            "    Unexpected key(s): {}".format(real_missing, unexpected_keys)
        )

    ignored_missing = [k for k in missing_keys if _is_dynamic_logcl_rel_emb_key(k)]
    if ignored_missing:
        print("[LogCL checkpoint] ignored dynamic missing rel_emb keys:", len(ignored_missing))
        for k in ignored_missing[:20]:
            print("  -", k)
        if len(ignored_missing) > 20:
            print("  ...")

    return incompatible


def _append_topk_score_records(records, score, triples, sample_id_start, direction, topk_k, query_embeddings=None):
    """Append LogCL top-k entity scores in the same sample order as DGS prompts.

    DGS builds each timestamp snapshot as all forward queries followed by all
    backward queries. LogCL evaluates the same test snapshot once in the forward
    direction and once after converting triples to inverse relations. The
    caller provides ``sample_id_start`` accordingly, so the exported ``id`` field
    aligns with the LLM prediction JSON and can be consumed directly by
    ``tools/hybrid_llm_logcl.py``.
    """
    if score is None or triples is None:
        return
    if torch.is_tensor(triples):
        triples_np = triples.detach().cpu().numpy()
    else:
        triples_np = np.asarray(triples)
    if len(triples_np) == 0:
        return

    score_cpu = score.detach().float().cpu()
    k = min(max(int(topk_k), 1), score_cpu.shape[1])
    top_values, top_indices = torch.topk(score_cpu, k=k, dim=1, largest=True, sorted=True)
    top_values = top_values.tolist()
    top_indices = top_indices.tolist()
    if query_embeddings is not None and torch.is_tensor(query_embeddings):
        query_embeddings = query_embeddings.detach().float().cpu().tolist()

    for row_idx, triple in enumerate(triples_np):
        sample_id = int(sample_id_start + row_idx)
        entity_ids = [int(x) for x in top_indices[row_idx]]
        entity_scores = [float(x) for x in top_values[row_idx]]
        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "direction": direction,
            "s_id": int(triple[0]),
            "r_id": int(triple[1]),
            "o_id": int(triple[2]),
            "topk_entity_ids": entity_ids,
            "topk_entity_scores": entity_scores,
            "scores": {str(eid): score for eid, score in zip(entity_ids, entity_scores)},
        }
        if query_embeddings is not None and row_idx < len(query_embeddings):
            row["query_embedding"] = [float(x) for x in query_embeddings[row_idx]]
        records.append(row)


def _save_topk_score_records(records, path, dataset, topk_k):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "dataset": dataset,
        "topk_score_k": int(topk_k),
        "score_type": "log_softmax",
        "query_embedding_type": "logcl_decoder_query",
        "scores": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Saved LogCL top-k scores for hybrid inference:", path)

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode, target_timestamps=None, score_split="test"):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    ranks_raw_r_inv, ranks_filter_r_inv, mrr_raw_list_r_inv, mrr_filter_list_r_inv = [], [], [], []
    ranks_raw1, ranks_filter1 = [],[]
    topk_score_records = []
    export_topk_scores = bool(getattr(args, "save_topk_scores", False)) and mode == "test"
    topk_score_k = int(getattr(args, "topk_score_k", args.topk))
    dense_writer = None
    export_dense_scores = bool(getattr(args, "save_dense_graph_scores", False)) and mode == "test"
    if export_dense_scores:
        dense_output_dir = getattr(args, "dense_graph_score_output_dir", None)
        if not dense_output_dir:
            dense_output_dir = os.path.join(args.result_dir, "graph_scores", args.dataset)
        dense_writer = DenseGraphScoreWriter(
            output_dir=dense_output_dir,
            split=score_split,
            num_queries=count_dense_graph_queries(test_list),
            num_entities=num_nodes,
        )
    snapshot_sample_offset = 0

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        map_location = torch.device(f"cuda:{args.gpu}") if use_cuda else torch.device("cpu")
        checkpoint = torch.load(model_name, map_location=map_location)
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        load_logcl_state_dict(model, checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    sr_to_sro = load_pickle_npy(os.path.join(args.data_root, args.dataset, 'his_dict', 'train_s_r.npy')).item()

    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples =test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        que_pair =  e2r(test_snap, num_rels)
        que_pair_inv =  e2r(inverse_triples, num_rels)

        sub_snap,sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap , num_nodes,num_rels,use_cuda, args.gpu)

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
        test_triples, final_score = model.predict(que_pair, sub_snap, time_idx, history_glist, num_rels, static_graph, test_triples_input, use_cuda)
        query_embeddings = getattr(model, "_last_query_embeddings", None)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)
        inv_query_embeddings = getattr(model, "_last_query_embeddings", None)

        if dense_writer is not None:
            timestamp = None
            if target_timestamps is not None and time_idx < len(target_timestamps):
                timestamp = int(target_timestamps[time_idx])
            for row_idx in range(test_triples.shape[0]):
                h = int(test_triples[row_idx, 0])
                r = int(test_triples[row_idx, 1])
                o = int(test_triples[row_idx, 2])
                dense_writer.write(
                    {
                        "id": f"{score_split}-{time_idx}-tail-{row_idx}",
                        "split": score_split,
                        "time_idx": int(time_idx),
                        "row_in_snapshot": int(row_idx),
                        "mode": "tail",
                        "query": [h, r, o, timestamp],
                        "known_entity": h,
                        "target_entity": o,
                        "query_relation_for_retrieval": r,
                    },
                    final_score[row_idx],
                )
            for row_idx in range(inv_test_triples.shape[0]):
                known_tail = int(inv_test_triples[row_idx, 0])
                inv_rel = int(inv_test_triples[row_idx, 1])
                target_head = int(inv_test_triples[row_idx, 2])
                orig_rel = int(inv_rel - num_rels)
                dense_writer.write(
                    {
                        "id": f"{score_split}-{time_idx}-head-{row_idx}",
                        "split": score_split,
                        "time_idx": int(time_idx),
                        "row_in_snapshot": int(row_idx),
                        "mode": "head",
                        "query": [target_head, orig_rel, known_tail, timestamp],
                        "known_entity": known_tail,
                        "target_entity": target_head,
                        "query_relation_for_retrieval": inv_rel,
                    },
                    inv_final_score[row_idx],
                )

        if export_topk_scores:
            _append_topk_score_records(
                topk_score_records,
                final_score,
                test_triples,
                snapshot_sample_offset,
                "forward",
                topk_score_k,
                query_embeddings,
            )
            _append_topk_score_records(
                topk_score_records,
                inv_final_score,
                inv_test_triples,
                snapshot_sample_offset + len(test_snap),
                "backward",
                topk_score_k,
                inv_query_embeddings,
            )

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
            # used to show slide results
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            # else:
            #     predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
            # subg_arr = np.concatenate([subg_arr,test_snap])
            # print(np.shape(subg_arr))
        snapshot_sample_offset += 2 * len(test_snap)
        idx += 1

    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)
    print("(all_raw) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))
    
    # 文件转储
    if mode == "test": # test模式写入，train模式忽略
        os.makedirs(args.result_dir, exist_ok=True)
        filename = os.path.join(args.result_dir, args.dataset + ".csv")
        if os.path.isfile(filename) == False:# 如果文件不存在，则创建
            with open (filename,'w', newline='') as f:
                # 写入列名
                fieldnames=['encoder','opn','pre_type','use_static','use_cl','gpu','datetime','pre_weight',
                            'train_len','test_len','temperature','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        # 写入数据
        with open (filename,'a', newline='') as f:
            writer = csv.writer(f)
            row={'encoder':args.encoder,'opn':args.opn,'pre_type':args.pre_type,'use_static':args.add_static_graph,'use_cl':args.use_cl,'gpu':args.gpu,'datetime':datetime.now(),'pre_weight':args.pre_weight,
                'train_len':args.train_history_len,'test_len':args.test_history_len,'temperature':args.temperature,'lr':args.lr,'n_hidden':args.n_hidden,
                'filter_MRR':float(mrr_filter),'filter_H@1':hit_filter[0],'filter_H@3':hit_filter[1],'filter_H@10':hit_filter[2],
                'filter_inv_MRR':float(mrr_filter_inv),'filter_inv_H@1':hit_filter_inv[0],'filter_inv_H@3':hit_filter_inv[1],'filter_inv_H@10':hit_filter_inv[2],
                'all_MRR':all_mrr_raw.item(),'all_H@1':all_hit_raw[0],'all_H@3':all_hit_raw[1],'all_H@10':all_hit_raw[2],
                'filter_all_MRR':all_mrr_filter.item(),'filter_all_H@1':all_hit_filter[0],'filter_all_H@3':all_hit_filter[1],'filter_all_H@10':all_hit_filter[2]}
            writer.writerow(row.values())

        if export_topk_scores:
            topk_score_file = getattr(args, "topk_score_file", None)
            if not topk_score_file:
                topk_score_file = os.path.join(args.result_dir, args.dataset + "_test_topk_scores.json")
            _save_topk_score_records(topk_score_records, topk_score_file, args.dataset, topk_score_k)
        if dense_writer is not None:
            dense_writer.close()
            print("Saved dense graph scores:", dense_writer.score_path)
            print("Saved dense graph queries:", dense_writer.query_path)
            print("Saved dense graph summary:", dense_writer.summary_path)
            
    return all_mrr_raw, all_mrr_filter
    

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset, data_root=args.data_root)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    train_timestamps = get_snapshot_timestamps(data.train)
    valid_timestamps = get_snapshot_timestamps(data.valid)
    test_timestamps = get_snapshot_timestamps(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    ensure_logcl_history_cache(
        args.dataset,
        args.data_root,
        train_list,
        num_rels,
        force_rebuild=getattr(args, "rebuild_logcl_cache", False),
    )

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_train = utils.load_all_answers_for_time_filter(data.train, num_rels, num_nodes, False)
    all_ans_list_r_train = utils.load_all_answers_for_time_filter(data.train, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature,args.pre_weight, args.use_cl, args.pre_type,  args.n_hidden, args.encoder,str(time.time()))
    if getattr(args, "model_state_file", None):
        model_state_file = os.path.abspath(args.model_state_file)
    else:
        model_state_file = os.path.join(args.model_dir, model_name + ".pt")
    os.makedirs(os.path.dirname(model_state_file) or '.', exist_ok=True)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(os.path.join(args.data_root, args.dataset, "e-w-graph.txt"), {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None


    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        pre_weight = args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type = args.pre_type,
                        use_cl = args.use_cl,
                        temperature = args.temperature,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        export_split = getattr(args, "export_score_split", "test")
        if export_split == "train":
            # DGS prompt training data skips the first train snapshot, then resets sample ids from 0.
            # Export the same target order so LogCL ids align with train.json.
            history_for_eval = train_list
            target_for_eval = train_list[1:]
            timestamps_for_eval = train_timestamps[1:]
            ans_for_eval = all_ans_list_train[1:]
            ans_r_for_eval = all_ans_list_r_train[1:]
        elif export_split == "valid":
            history_for_eval = train_list
            target_for_eval = valid_list
            timestamps_for_eval = valid_timestamps
            ans_for_eval = all_ans_list_valid
            ans_r_for_eval = all_ans_list_r_valid
        else:
            history_for_eval = train_list + valid_list
            target_for_eval = test_list
            timestamps_for_eval = test_timestamps
            ans_for_eval = all_ans_list_test
            ans_r_for_eval = all_ans_list_r_test
        mrr_raw, mrr_filter= test(model,
                                history_for_eval, 
                                target_for_eval, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                ans_for_eval, 
                                ans_r_for_eval, 
                                model_state_file, 
                                static_graph, 
                                "test",
                                target_timestamps=timestamps_for_eval,
                                score_split=export_split)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = -1.0
        his_best = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                subgraph_arr = np.load(os.path.join(args.data_root, args.dataset, 'his_graph_for', 'train_s_r_{}.npy'.format(train_sample_num)))
                subgraph_arr_inv = np.load(os.path.join(args.data_root, args.dataset, 'his_graph_inv', 'train_o_r_{}.npy'.format(train_sample_num)))
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)   #取出采样子图
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair =  e2r(output[0], num_rels)
                que_pair_inv =  e2r(inverse_triples, num_rels)
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda() 
                for id in range(2): 
                    if id %2 ==0: 
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair, subg_snap, train_sample_num, history_glist, triples, static_graph, use_cuda)
                    else:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num, history_glist, inverse_triples,static_graph, use_cuda)

                    loss = loss_e+ loss_static +loss_cl
                
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                # break
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, 
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph, 
                                    mode="train",
                                    target_timestamps=valid_timestamps,
                                    score_split="valid")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            break
                        if his_best>=5:
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': clean_logcl_state_dict(model.state_dict()), 'epoch': epoch, 'args': vars(args)}, model_state_file)
            torch.cuda.empty_cache()
        if not os.path.exists(model_state_file):
            fallback_epoch = args.n_epochs - 1
            print("Warning: no validation checkpoint was saved; saving the last graph model to {}".format(model_state_file))
            torch.save({'state_dict': clean_logcl_state_dict(model.state_dict()), 'epoch': fallback_epoch, 'args': vars(args)}, model_state_file)
        mrr_raw, mrr_filter = test(model, 
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph, 
                            mode="test",
                            target_timestamps=test_timestamps,
                            score_split="test")
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogCL')

    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="short",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=True,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--model-state-file", type=str, default=None,
                        help="optional stable checkpoint path used for both saving and testing")
    parser.add_argument("--data-root", type=str, default="../data/logcl",
                        help="root directory containing <dataset>/train.txt and LogCL history cache")
    parser.add_argument("--model-dir", type=str, default="../outputs/graph",
                        help="directory for fallback LogCL checkpoint files")
    parser.add_argument("--result-dir", type=str, default="../outputs/graph/logcl_result",
                        help="directory for LogCL evaluation CSV files")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")
    parser.add_argument("--save-topk-scores", action="store_true", default=True,
                        help="save LogCL top-k entity scores for LLM+LogCL hybrid inference")
    parser.add_argument("--topk-score-file", type=str, default=None,
                        help="path to save LogCL top-k score JSON")
    parser.add_argument("--topk-score-k", type=int, default=50,
                        help="number of LogCL graph candidates saved for each query")
    parser.add_argument("--export-score-split", type=str, default="test", choices=["train", "valid", "test"],
                        help="which split to export when --test --save-topk-scores is enabled")
    parser.add_argument("--save-dense-graph-scores", action="store_true", default=True,
                        help="save dense graph score matrix for every query and every entity")
    parser.add_argument("--dense-graph-score-output-dir", type=str, default='./output/graph_score',
                        help="directory for dense graph score files; default: <result-dir>/graph_scores/<dataset>")
    parser.add_argument("--rebuild-logcl-cache", action="store_true", default=False,
                        help="rebuild LogCL his_graph_for/his_graph_inv/his_dict cache before training or testing")


    args = parser.parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.model_dir = os.path.abspath(args.model_dir)
    args.result_dir = os.path.abspath(args.result_dir)
    if args.dense_graph_score_output_dir:
        args.dense_graph_score_output_dir = os.path.abspath(args.dense_graph_score_output_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    print(args)
    # Keep the user-provided --test-history-len.  The original release
    # overwrote it with --train-history-len, which made graph-only
    # inference hard to reproduce from an external checkpoint.

    run_experiment(args)
