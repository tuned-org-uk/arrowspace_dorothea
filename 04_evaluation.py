#!/usr/bin/env python3
"""
Final Evaluation: Dorothea Test Queries vs ArrowSpace Index.
Uses the new load_arrowspace method to restore persistent indices from Parquet.
Performs seeded densification on test queries and runs spectral search.
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
from arrowspace import load_arrowspace

import logging
logging.basicConfig(level=logging.INFO)

# 1. Seeded Densification (Matches Build Pipeline)
def densify_seeded(X_bin: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Injects deterministic noise and L2-normalizes to match build geometry."""
    rng = np.random.default_rng(seed)
    X = X_bin.astype(np.float64) + rng.normal(0.0, noise_level, size=X_bin.shape)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

def read_sparse_binary(path: Path, n_features: int) -> np.ndarray:
    """Parses Dorothea sparse text format."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            idx = [int(t) - 1 for t in line.strip().split() if t]
            x = np.zeros(n_features)
            x[idx] = 1.0
            rows.append(x)
    return np.stack(rows)

# 2. Metric suite
def compute_metrics(results_a, results_b, k=10):
    """Computes Spearman, Kendall, and NDCG proxy between two rankings."""
    idx_a = [i for i, _ in results_a]
    idx_b = [i for i, _ in results_b]
    
    shared = list(set(idx_a).intersection(idx_b))
    if len(shared) >= 2:
        ra = [idx_a.index(i) for i in shared]
        rb = [idx_b.index(i) for i in shared]
        s, _ = spearmanr(ra, rb)
        k_val, _ = kendalltau(ra, rb)
    else:
        s, k_val = 0.0, 0.0

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
    storage_dir = Path(args.storage).resolve()
    
    # A. Initial check of dataset availability
    if not (storage_dir / f"{args.dataset}-raw_input.parquet").exists():
        logging.error(f"Dataset {args.dataset} not found in {storage_dir}")
        return

    # B. Load ArrowSpace from storage (No recomputation)
    # This automatically restores:
    # - ImplicitProjections (seeds/dims)
    # - Sorted Lambda Index
    # - Graph Laplacian (init_data + sparse matrix)
    graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
    
    logging.info(f"Restoring ArrowSpace index for: {args.dataset}")
    aspace, gl = load_arrowspace(
        storage_path=str(storage_dir),
        dataset_name=args.dataset,
        graph_params=graph_params,
        energy=False,
    )

    logging.info(f"Successfully Loaded: {aspace.nitems} items × {aspace.nfeatures} features")
    logging.info(f"Graph Nodes: {gl.nnodes}")

    # C. Load & Densify Test Queries
    # We use the feature dimension from the loaded index
    X_test_raw = read_sparse_binary(Path(args.test_data), aspace.nfeatures)
    X_test = densify_seeded(X_test_raw, noise_level=0.001, seed=42)
    logging.info(f"Test queries densified: {X_test.shape}")

    # D. Evaluation Loop
    tau_configs = {"Cosine": 1.0, "Hybrid": 0.72, "TauMode": 0.42}
    results_summary = []

    for i in tqdm(range(min(len(X_test), args.n_queries)), desc="Evaluating"):
        q = X_test[i]
        test_completed = 0
        zeroed_test = []

        # Use the real search API
        # If the index was reduced (JL), the search method internally reprojects 'q' 
        # using the restored seed from arrowspace_metadata.json
        try:
            r_cos = aspace.search(q, gl, tau=tau_configs["Cosine"])
            r_hyb = aspace.search(q, gl, tau=tau_configs["Hybrid"])
            r_tau = aspace.search(q, gl, tau=tau_configs["TauMode"])
            test_completed += 1
        except:
            print(f"test query {i} got lambda == 0.0")
            zeroed_test.append({
            "query_idx": i,
            "vector": q
            })
            continue

        # Metrics vs Cosine Baseline
        m_hyb = compute_metrics(r_hyb, r_cos)
        m_tau = compute_metrics(r_tau, r_cos)

        results_summary.append({
            "query_idx": i,
            "hybrid_ndcg": m_hyb["ndcg"],
            "taumode_ndcg": m_tau["ndcg"],
            "taumode_spearman": m_tau["spearman"]
        })

    # E. Report and Save
    df = pd.DataFrame(results_summary)
    print("\n[Mean Evaluation Metrics]")
    print(df[["hybrid_ndcg", "taumode_ndcg", "taumode_spearman"]].mean())
    
    output_path = storage_dir / f"{args.dataset}_eval_results.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    
    # stored zeroed queries
    output_path_zeroed = storage_dir / f"{args.dataset}_eval_zeroed.csv"
    df_zeroed = pd.DataFrame(zeroed_test)
    df_zeroed.to_csv(output_path_zeroed, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--storage", default="./storage", help="Directory with Parquet/JSON files")
    p.add_argument("--dataset", default="dorothea_highdim", help="Prefix for stored files")
    p.add_argument("--test-data", default="data/DOROTHEA/dorothea_test.data")
    p.add_argument("--n-queries", type=int, default=50)
    main(p.parse_args())
