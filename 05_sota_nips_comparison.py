#!/usr/bin/env python3
"""
ArrowSpace k-NN Classification Evaluation on Dorothea.
Compares spectral-aware search against vanilla cosine k-NN using
the official NIPS 2003 benchmark metric: Balanced Error Rate (BER).

UPDATED: 
- Uses `load_arrowspace` to restore pre-computed index.
- Correctly handles the 1150-item index (Train + Valid) by concatenating labels.
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
    """
    Majority-vote k-NN classification via ArrowSpace search.
    The `train_labels` array must match the index size (1150 items).
    """
    # aspace.search returns indices [0..1149]
    results = aspace.search(query, gl, tau=tau)[:k]
    
    # Map indices to labels. Since our index is (Train + Valid), 
    # train_labels must be the concatenated (y_train + y_valid).
    neighbor_labels = [train_labels[idx] for idx, _ in results]
    
    return int(np.sign(np.sum(neighbor_labels)))

def knn_predict_cosine_baseline(X_ref, y_ref, query, k: int) -> int:
    """Pure cosine k-NN (no ArrowSpace) for baseline comparison."""
    from scipy.spatial.distance import cdist
    dists = cdist(query.reshape(1, -1), X_ref, metric="cosine")[0]
    topk_idx = np.argpartition(dists, k)[:k]
    neighbor_labels = y_ref[topk_idx]
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "BER": float(ber),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
    }

# ============================================================================
# 4. EVALUATION PIPELINE
# ============================================================================

def evaluate_classification(args):
    data_dir = Path(args.data_dir)
    storage_dir = Path(args.storage).resolve()
    
    print("="*70)
    print("ArrowSpace k-NN Classification Evaluation on Dorothea")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # A. Restore ArrowSpace Index
    # -------------------------------------------------------------------------
    logging.info(f"Loading ArrowSpace Index: {args.dataset}")
    
    # Use params that match the build
    graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
    
    aspace, gl = load_arrowspace(
        storage_path=str(storage_dir),
        dataset_name=args.dataset,
        graph_params=graph_params,
        energy=False,
    )
    logging.info(f"Loaded: {aspace.nitems} items × {aspace.nfeatures} features")
    logging.info(f"Graph Nodes: {gl.nnodes}")

    # -------------------------------------------------------------------------
    # B. Load & Align Labels (The Critical Fix)
    # -------------------------------------------------------------------------
    logging.info("Loading labels for Training + Validation sets...")
    
    y_train = read_labels(data_dir / "dorothea_train.labels")
    y_valid = read_labels(data_dir / "dorothea_valid.labels")
    
    # IMPORTANT: The loaded index contains 1150 items (800 Train + 350 Valid).
    # We must concatenate the labels in the exact same order (Train then Valid).
    y_all = np.concatenate([y_train, y_valid])
    
    if len(y_all) != aspace.nitems:
        logging.error(f"Label mismatch! Index has {aspace.nitems} items, but loaded {len(y_all)} labels.")
        return

    # -------------------------------------------------------------------------
    # C. Prepare Baseline Data (X_all)
    # -------------------------------------------------------------------------
    logging.info("Loading raw data for baseline comparison...")
    
    X_train_raw = read_sparse_binary(data_dir / "dorothea_train.data", aspace.nfeatures)
    X_valid_raw = read_sparse_binary(data_dir / "dorothea_valid.data", aspace.nfeatures)
    X_all_raw = np.vstack([X_train_raw, X_valid_raw])
    
    # Apply same densification as index
    X_all = densify_seeded(X_all_raw, noise_level=args.noise, seed=args.seed)

    # -------------------------------------------------------------------------
    # D. Load Test Data
    # -------------------------------------------------------------------------
    logging.info("Loading test data...")
    test_data_path = data_dir / "dorothea_test.data"
    test_labels_path = data_dir / "dorothea_test.labels"
    
    if test_data_path.exists() and test_labels_path.exists():
        logging.info("Using official test split")
        X_test_raw = read_sparse_binary(test_data_path, aspace.nfeatures)
        y_test = read_labels(test_labels_path)
    else:
        raise ValueError(f"Missing {test_data_path} or {test_labels_path}")
    
    X_test = densify_seeded(X_test_raw, noise_level=args.noise, seed=args.seed)[:50]
    logging.info(f"Test Set: {len(X_test)} samples")

    # -------------------------------------------------------------------------
    # E. Experiments
    # -------------------------------------------------------------------------
    tau_configs = {
        "Cosine (τ=1.0)": 1.0,
        "Hybrid (τ=0.72)": 0.72,
        "TauMode (τ=0.42)": 0.42,
    }
    
    k_values = args.k_values
    results = []
    
    # 1. Spectral-Aware k-NN
    for method_name, tau in tau_configs.items():
        for k in k_values:
            logging.info(f"Evaluating {method_name}, k={k}")
            
            predictions = []
            for q in tqdm(X_test, desc=f"{method_name} k={k}", leave=False):
                # We pass y_all (1150 labels) to match the index indices
                pred = knn_predict(aspace, gl, q, k=k, tau=tau, train_labels=y_all)
                predictions.append(pred)
            
            metrics = compute_ber(y_test, np.array(predictions))
            results.append({"method": method_name, "tau": tau, "k": k, **metrics})

    # 2. Pure Cosine Baseline (Searching against X_all)
    for k in k_values:
        logging.info(f"Evaluating Baseline Cosine, k={k}")
        
        predictions = []
        for q in tqdm(X_test, desc=f"Baseline k={k}", leave=False):
            # Baseline searches against the same 1150 items
            pred = knn_predict_cosine_baseline(X_all, y_all, q, k=k)
            predictions.append(pred)
        
        metrics = compute_ber(y_test, np.array(predictions))
        results.append({"method": "Baseline Cosine", "tau": 1.0, "k": k, **metrics})

    # -------------------------------------------------------------------------
    # F. Save & Report
    # -------------------------------------------------------------------------
    df = pd.DataFrame(results)
    df = df.sort_values(["method", "k"])
    
    out_csv = storage_dir / f"{args.dataset}_classification_results.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Results saved to {out_csv}")
    
    print("\n[Best Results per Method (Lowest BER)]")
    best_per_method = df.loc[df.groupby("method")["BER"].idxmin()]
    print(best_per_method[["method", "k", "BER", "precision", "recall", "f1"]].to_string(index=False))

    # Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in df["method"].unique():
            subset = df[df["method"] == method]
            ax.plot(subset["k"], subset["BER"] * 100, marker="o", label=method)
        
        ax.set_xlabel("k (Neighbors)")
        ax.set_ylabel("Balanced Error Rate (%)")
        ax.set_title("Classification Performance: BER vs k")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(storage_dir / f"{args.dataset}_ber_plot.png")
        logging.info("Saved BER plot")
    except Exception as e:
        logging.warning(f"Could not generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/DOROTHEA")
    parser.add_argument("--storage", default="./storage")
    parser.add_argument("--dataset", default="dorothea_highdim") # The prefix of your stored files
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 15, 25])
    
    evaluate_classification(parser.parse_args())
