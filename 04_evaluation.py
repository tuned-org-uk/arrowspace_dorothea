#!/usr/bin/env python3
"""
Final Evaluation: Dorothea Test Queries vs ArrowSpace Index.
Performs seeded densification on test queries and runs spectral search
using the real ArrowSpace search API to compute query-time TauMode.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
from arrowspace import ArrowSpaceBuilder

# 1. Seeded Densification (Matches Build Pipeline)
def densify_seeded(X_bin: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Injects deterministic noise and L2-normalizes to match build geometry.""" # [file:48]
    rng = np.random.default_rng(seed)
    X = X_bin.astype(np.float64) + rng.normal(0.0, noise_level, size=X_bin.shape)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

def read_sparse_binary(path: Path, n_features: int) -> np.ndarray:
    """Parses Dorothea sparse text format.""" # [file:4]
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            idx = [int(t) - 1 for t in line.strip().split() if t]
            x = np.zeros(n_features)
            x[idx] = 1.0
            rows.append(x)
    return np.stack(rows)

# 2. Paper-Style Metric suite (CVE Test Methodology)
def compute_metrics(results_a, results_b, k=10):
    """Computes Spearman, Kendall, and NDCG proxy between two rankings.""" # [file:46]
    idx_a = [i for i, _ in results_a]
    idx_b = [i for i, _ in results_b]
    
    # Spearman/Kendall on shared IDs
    shared = list(set(idx_a).intersection(idx_b))
    if len(shared) >= 2:
        ra = [idx_a.index(i) for i in shared]
        rb = [idx_b.index(i) for i in shared]
        s, _ = spearmanr(ra, rb)
        k_val, _ = kendalltau(ra, rb)
    else:
        s, k_val = 0.0, 0.0

    # NDCG@k treating results_b as ground truth reference
    ref_idx = [i for i, _ in results_b[:k]]
    rel_map = {idx: (k - r) for r, idx in enumerate(ref_idx)}
    pred_idx = [i for i, _ in results_a[:k]]
    true_rel = np.array([rel_map.get(i, 0) for i in pred_idx])
    
    if true_rel.sum() > 0:
        pred_scores = np.array([s for _, s in results_a[:k]])
        ndcg = ndcg_score(true_rel.reshape(1, -1), pred_scores.reshape(1, -1), k=k)
    else:
        ndcg = 0.0
        
    return {"spearman": float(s), "kendall": float(k_val), "ndcg": float(ndcg)}

def main(args):
    storage = Path(args.storage)
    
    # A. Load Build Data
    X_index = np.load(storage / "dorothea_highdim_full100k.npy")
    print(f"Index loaded: {X_index.shape}")

    # B. Re-Build/Initialize ArrowSpace for search
    graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
    aspace, gl = ArrowSpaceBuilder().build_full(graph_params, X_index)

    # C. Load & Densify Test Queries
    X_test_raw = read_sparse_binary(Path(args.test_data), X_index.shape[1])
    X_test = densify_seeded(X_test_raw, noise_level=0.001, seed=42)
    print(f"Test queries densified: {X_test.shape}")

    # D. Multi-Tau Evaluation
    tau_configs = {"Cosine": 1.0, "Hybrid": 0.72, "TauMode": 0.42}
    results_summary = []

    for i in tqdm(range(min(len(X_test), args.n_queries)), desc="Querying"):
        q = X_test[i]
        
        # Search calls: TauMode is automatically computed here [file:46]
        r_cos = aspace.search(q, gl, tau=tau_configs["Cosine"])
        r_hyb = aspace.search(q, gl, tau=tau_configs["Hybrid"])
        r_tau = aspace.search(q, gl, tau=tau_configs["TauMode"])

        # Metrics vs Cosine Baseline
        m_hyb = compute_metrics(r_hyb, r_cos)
        m_tau = compute_metrics(r_tau, r_cos)

        results_summary.append({
            "query_idx": i,
            "hybrid_ndcg": m_hyb["ndcg"],
            "taumode_ndcg": m_tau["ndcg"],
            "taumode_spearman": m_tau["spearman"]
        })

    # E. Report
    df = pd.DataFrame(results_summary)
    print("\n[Summary Statistics]")
    print(df[["hybrid_ndcg", "taumode_ndcg", "taumode_spearman"]].mean())
    
    # Save results
    df.to_csv(storage / "dorothea_eval_results.csv", index=False)
    print(f"\nResults saved to {storage / 'dorothea_eval_results.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--storage", default="./storage")
    p.add_argument("--test-data", default="data/DOROTHEA/dorothea_test.data")
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--debug", action="store_true")
    main(p.parse_args())
