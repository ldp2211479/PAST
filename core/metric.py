from typing import Dict, List


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def compute_rank(pred_list: List[str], ground_truth: str, filters: List[str]) -> float:
    gt_norm = normalize_text(ground_truth)
    filter_entities = {normalize_text(item) for item in filters}
    if gt_norm in filter_entities:
        filter_entities.remove(gt_norm)

    rank = 1
    for pred in pred_list:
        pred_norm = normalize_text(pred)
        if pred_norm == gt_norm:
            return rank
        if pred_norm in filter_entities:
            continue
        rank += 1
    return float("inf")


def compute_metrics(all_preds: List[List[str]], all_labels: List[str], all_filters: List[List[str]]) -> Dict[str, float]:
    metrics = {
        "mrr": 0.0,
        "hits@1": 0.0,
        "hits@3": 0.0,
        "hits@10": 0.0,
        "hits@30": 0.0,
        "hits@50": 0.0,
    }

    total = len(all_preds)
    for preds, label, filters in zip(all_preds, all_labels, all_filters):
        rank = compute_rank(preds, label, filters)
        if rank <= 1:
            metrics["hits@1"] += 1
        if rank <= 3:
            metrics["hits@3"] += 1
        if rank <= 10:
            metrics["hits@10"] += 1
        if rank <= 30:
            metrics["hits@30"] += 1
        if rank <= 50:
            metrics["hits@50"] += 1
        if rank != float("inf"):
            metrics["mrr"] += 1.0 / rank

    if total == 0:
        return metrics

    for key in metrics:
        metrics[key] /= total
    return metrics
