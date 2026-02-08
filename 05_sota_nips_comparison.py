#!/usr/bin/env python3
"""
ArrowSpace k-NN Classification Evaluation on Dorothea.
Compares spectral-aware search against vanilla cosine k-NN using
the official NIPS 2003 benchmark metric: Balanced Error Rate (BER).
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from arrowspace import load_arrowspace

import logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# 1. DATA LOADING (Dorothea Benchmark Format)
# ============================================================================

def read_sparse_binary(path: Path, n_features: int) -> np.ndarray:
    """Parse Dorothea sparse-binary format: 1-based indices."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            idx = [int(t) - 1 for t in line.strip().split() if t]
            x = np.zeros(n_features, dtype=np.float64)
            x[idx] = 1.0
            rows.append(x)
    return np.stack(rows, axis=0)

def read_labels(path: Path) -> np.ndarray:
    """Load labels: +1 (Active) or -1 (Inactive)."""
    return np.loadtxt(path, dtype=np.int32)

def densify_seeded(X_bin: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Apply seeded Gaussian noise + L2 normalization (matches build pipeline)."""
    rng = np.random.default_rng(seed)
    X = X_bin.astype(np.float64) + rng.normal(0.0, noise_level, size=X_bin.shape)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

# ============================================================================
# 2. k-NN CLASSIFICATION
# ============================================================================

def knn_predict(aspace, gl, query, k: int, tau: float, train_labels: np.ndarray) -> int:
    """Majority-vote k-NN classification via ArrowSpace search."""
    # aspace.search internally handles query reprojection if JL was used during build
    results = aspace.search(query, gl, tau=tau)[:k]
    neighbor_labels = [train_labels[idx] for idx, _ in results]
    return int(np.sign(np.sum(neighbor_labels)))

def knn_predict_cosine_baseline(X_train, y_train, query, k: int) -> int:
    """Pure cosine k-NN (no ArrowSpace) for baseline comparison."""
    from scipy.spatial.distance import cdist
    dists = cdist(query.reshape(1, -1), X_train, metric="cosine")[0]
    topk_idx = np.argpartition(dists, k)[:k]
    neighbor_labels = y_train[topk_idx]
    return int(np.sign(np.sum(neighbor_labels)))

# ============================================================================
# 3. BALANCED ERROR RATE (Official NIPS 2003 Metric)
# ============================================================================

def compute_ber(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Balanced Error Rate (BER) = 0.5 * (FP/(FP+TN) + FN/(FN+TP))."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == -1)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    tp = ((y_pred == 1) & pos_mask).sum()
    fn = ((y_pred == -1) & pos_mask).sum()
    tn = ((y_pred == -1) & neg_mask).sum()
    fp = ((y_pred == 1) & neg_mask).sum()
    
    pos_error = fn / n_pos if n_pos > 0 else 0.0
    neg_error = fp / n_neg if n_neg > 0 else 0.0
    ber = 0.5 * (pos_error + neg_error)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "BER": float(ber),
        "precision": float(precision),
        "recall": float(recall),
        "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
    }

# ============================================================================
# 4. EVALUATION PIPELINE
# ============================================================================

def evaluate_classification(args):
    data_dir = Path(args.data_dir)
    storage_dir = Path(args.storage).resolve()
    
    logging.info(f"Loading ArrowSpace Index: {args.dataset}")
    
    # A. Restore ArrowSpace (Automatically handles projection restoration)
    # Using parameters that match the build phase
    graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
    aspace, gl = load_arrowspace(
        storage_path=str(storage_dir),
        dataset_name=args.dataset,
        graph_params=graph_params,
        energy=False,
    )
    logging.info(f"Loaded: {aspace.nitems} training items, {aspace.nfeatures} features")

    # B. Load Training Labels & Raw Data (for baseline)
    y_train = read_labels(data_dir / "dorothea_train.labels")
    X_train_raw = read_sparse_binary(data_dir / "dorothea_train.data", aspace.nfeatures)
    X_train = densify_seeded(X_train_raw, noise_level=args.noise, seed=args.seed)

    # C. Load & Densify Test/Validation Data
    test_data_path = data_dir / "dorothea_test.data"
    test_labels_path = data_dir / "dorothea_test.labels"
    
    if test_data_path.exists() and test_labels_path.exists():
        logging.info("Using official test split")
        X_test_raw = read_sparse_binary(test_data_path, aspace.nfeatures)
        y_test = read_labels(test_labels_path)
    else:
        logging.info("Official test labels not found, using validation split")
        X_test_raw = read_sparse_binary(data_dir / "dorothea_valid.data", aspace.nfeatures)
        y_test = read_labels(data_dir / "dorothea_valid.labels")
    
    X_test = densify_seeded(X_test_raw, noise_level=args.noise, seed=args.seed)

    # D. Experiments
    tau_configs = {
        "Cosine (τ=1.0)": 1.0,
        "Hybrid (τ=0.72)": 0.72,
        "TauMode (τ=0.42)": 0.42,
    }
    
    results = []
    k_values = [5, 10, 15, 20, 25]
    
    # 1. Spectral-Aware k-NN
    for method_name, tau in tau_configs.items():
        for k in k_values:
            logging.info(f"Evaluating {method_name}, k={k}")
            preds = [knn_predict(aspace, gl, q, k, tau, y_train) for q in tqdm(X_test)]
            metrics = compute_ber(y_test, np.array(preds))
            results.append({"method": method_name, "tau": tau, "k": k, **metrics})

    # 2. Pure Cosine Baseline
    for k in k_values:
        logging.info(f"Evaluating Baseline Cosine, k={k}")
        preds = [knn_predict_cosine_baseline(X_train, y_train, q, k) for q in tqdm(X_test)]
        metrics = compute_ber(y_test, np.array(preds))
        results.append({"method": "Baseline Cosine", "tau": 1.0, "k": k, **metrics})

    # E. Report Results
    df = pd.DataFrame(results)
    out_csv = storage_dir / f"{args.dataset}_classification_results.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n[Best Results per Method]")
    print(df.loc[df.groupby("method")["BER"].idxmin()][["method", "k", "BER", "precision", "recall"]])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/DOROTHEA")
    p.add_argument("--storage", default="./storage")
    p.add_argument("--dataset", default="dorothea_highdim")
    p.add_argument("--noise", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    evaluate_classification(p.parse_args())
