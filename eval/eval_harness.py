"""Local evaluation harness — Blueprint v8."""

from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np


def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size + 1)
    return np.sum(relevances / np.log2(positions + 1))


def ndcg_at_k(ranked_ids, relevant_ids_with_scores, k):
    rel_map = dict(relevant_ids_with_scores)
    actual = [rel_map.get(cid, 0) for cid in ranked_ids[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(actual, k) / ideal_dcg


def average_precision(ranked_ids, relevant_ids):
    hits, score = 0, 0.0
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_ids:
            hits += 1
            score += hits / i
    return score / max(len(relevant_ids), 1)


def precision_at_k(ranked_ids, relevant_ids, k):
    return sum(1 for cid in ranked_ids[:k] if cid in relevant_ids) / k


def full_score(ranked_ids: Sequence[str], pseudo_labels: dict[str, float]):
    """Compute the exact spec formula + tiebreaker metrics."""
    relevant = {cid for cid, score in pseudo_labels.items() if score >= 6}
    rel_with_scores = [(cid, s) for cid, s in pseudo_labels.items()]

    ndcg_10 = ndcg_at_k(ranked_ids, rel_with_scores, 10)
    ndcg_50 = ndcg_at_k(ranked_ids, rel_with_scores, 50)
    map_score = average_precision(ranked_ids, relevant)
    p_at_10 = precision_at_k(ranked_ids, relevant, 10)
    p_at_5 = precision_at_k(ranked_ids, relevant, 5)

    composite = 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_score + 0.05 * p_at_10

    print(f"NDCG@10:  {ndcg_10:.4f}  (weight 50%)")
    print(f"NDCG@50:  {ndcg_50:.4f}  (weight 30%)")
    print(f"MAP:      {map_score:.4f}  (weight 15%)")
    print(f"P@10:     {p_at_10:.4f}  (weight  5%  | tiebreaker #2)")
    print(f"P@5:      {p_at_5:.4f}  (tiebreaker #1 — optimize this first!)")
    print(f"COMPOSITE:{composite:.4f}")
    return composite, p_at_5, p_at_10


def evaluate_csv(csv_path: str, pseudo_labels_path: str = "pseudo_labels.pkl"):
    import csv

    ranked_rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranked_rows.append(row)

    ranked_ids = [row["candidate_id"] for row in ranked_rows]
    pseudo_labels = pickle.load(open(pseudo_labels_path, "rb"))
    composite, p5, p10 = full_score(ranked_ids, pseudo_labels)
    print(f"\nReady to submit? Composite={composite:.4f}, P@5={p5:.4f}, P@10={p10:.4f}")
    return composite, p5, p10


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ranked output CSV from rank.py")
    parser.add_argument("--pseudo-labels", default="pseudo_labels.pkl")
    args = parser.parse_args()
    evaluate_csv(args.csv, args.pseudo_labels)
