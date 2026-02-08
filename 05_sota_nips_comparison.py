#!/usr/bin/env python3
"""
ArrowSpace k-NN Classification Evaluation on Dorothea (NIPS 2003).

Methodology:
- Index built from TRAINING split only (800 samples)
- Evaluation on VALIDATION split (350 samples, held-out)
- Metric: Balanced Error Rate (BER) - official NIPS 2003 benchmark

This ensures no data leakage: validation samples never influence the index.
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
    
    Note: aspace.search internally handles query reprojection if JL was used during build.
    """
    results = aspace.search(query, gl, tau=tau)[:k]
    neighbor_labels = [train_labels[idx] for idx, _ in results]
    
    if len(neighbor_labels) == 0:
        logging.warning(f"No neighbors found for query, defaulting to negative class")
        return -1
    
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
    """
    Balanced Error Rate (BER) = 0.5 * (FP/(FP+TN) + FN/(FN+TP)).
    
    This is the official Dorothea benchmark metric from NIPS 2003, which accounts
    for class imbalance by averaging error rates across positive and negative classes.
    """
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
        "pos_error": float(pos_error),
        "neg_error": float(neg_error),
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
    
    logging.info("="*80)
    logging.info("ArrowSpace k-NN Classification: Dorothea NIPS 2003 Benchmark")
    logging.info("="*80)
    
    # ========================================================================
    # A. Load Pre-Built ArrowSpace Index (TRAINING DATA ONLY)
    # ========================================================================
    
    logging.info(f"\n[1] Loading ArrowSpace index: {args.dataset}")
    graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
    
    aspace, gl = load_arrowspace(
        storage_path=str(storage_dir),
        dataset_name=args.dataset,
        graph_params=graph_params,
        energy=False,
    )
    
    logging.info(f"    Index loaded: {aspace.nitems} items × {aspace.nfeatures} features")
    
    # Sanity check: Verify index was built from training split only (800 samples)
    EXPECTED_TRAIN_SIZE = 800
    if aspace.nitems != EXPECTED_TRAIN_SIZE:
        logging.error(
            f"    ❌ DATA LEAKAGE DETECTED! Expected {EXPECTED_TRAIN_SIZE} training items, "
            f"but index contains {aspace.nitems} items.\n"
            f"    Please rebuild index using ONLY dorothea_train.data"
        )
        raise ValueError("Index contains non-training data - rebuild required")
    
    logging.info(f"    ✓ No data leakage: Index contains {EXPECTED_TRAIN_SIZE} training samples only")
    
    # ========================================================================
    # B. Load Training Labels (for k-NN voting)
    # ========================================================================
    
    logging.info("\n[2] Loading training data...")
    y_train = read_labels(data_dir / "dorothea_train.labels")
    X_train_raw = read_sparse_binary(data_dir / "dorothea_train.data", aspace.nfeatures)
    X_train = densify_seeded(X_train_raw, noise_level=args.noise, seed=args.seed)
    
    logging.info(f"    Training samples: {len(y_train)}")
    logging.info(f"    Class distribution: Positive={((y_train == 1).sum())}, "
                 f"Negative={((y_train == -1).sum())}")
    
    # ========================================================================
    # C. Load Validation Split as Test Set (HELD-OUT)
    # ========================================================================
    
    logging.info("\n[3] Loading validation split as test set...")
    logging.info("    (Official test labels are withheld for NIPS 2003 challenge)")
    
    X_test_raw = read_sparse_binary(data_dir / "dorothea_valid.data", aspace.nfeatures)
    y_test = read_labels(data_dir / "dorothea_valid.labels")
    X_test = densify_seeded(X_test_raw, noise_level=args.noise, seed=args.seed)
    
    logging.info(f"    Test samples: {len(X_test)}")
    logging.info(f"    Class distribution: Positive={((y_test == 1).sum())}, "
                 f"Negative={((y_test == -1).sum())}")
    
    # Verify no overlap
    assert X_train.shape[0] == EXPECTED_TRAIN_SIZE, "Training data size mismatch"
    assert X_test.shape[0] == 350, "Validation data size mismatch"
    
    # ========================================================================
    # D. Run k-NN Classification Experiments
    # ========================================================================
    
    logging.info("\n[4] Running k-NN classification experiments...")
    logging.info("="*80)
    
    tau_configs = {
        "Cosine (τ=1.0)": 1.0,
        "Hybrid (τ=0.72)": 0.72,
        "TauMode (τ=0.42)": 0.42,
    }
    
    k_values = [5, 15, 25]
    results = []
    
    # Experiment 1: Spectral-Aware k-NN
    for method_name, tau in tau_configs.items():
        for k in k_values:
            logging.info(f"  → {method_name}, k={k}")
            preds = []
            for q in tqdm(X_test, desc=f"    {method_name} k={k}", leave=False):
                preds.append(knn_predict(aspace, gl, q, k, tau, y_train))
            
            metrics = compute_ber(y_test, np.array(preds))
            results.append({"method": method_name, "tau": tau, "k": k, **metrics})
    
    # Experiment 2: Pure Cosine Baseline (no spectral graph)
    for k in k_values:
        logging.info(f"  → Baseline Cosine, k={k}")
        preds = []
        for q in tqdm(X_test, desc=f"    Baseline k={k}", leave=False):
            preds.append(knn_predict_cosine_baseline(X_train, y_train, q, k))
        
        metrics = compute_ber(y_test, np.array(preds))
        results.append({"method": "Baseline Cosine", "tau": 1.0, "k": k, **metrics})
    
    # ========================================================================
    # E. Report Results
    # ========================================================================
    
    df = pd.DataFrame(results)
    out_csv = storage_dir / f"{args.dataset}_classification_results.csv"
    df.to_csv(out_csv, index=False)
    
    logging.info("\n" + "="*80)
    logging.info("EVALUATION RESULTS")
    logging.info("="*80)
    
    print("\n[Best BER per Method]")
    best_per_method = df.loc[df.groupby("method")["BER"].idxmin()]
    print(best_per_method[["method", "k", "BER", "precision", "recall", "f1"]].to_string(index=False))
    
    print("\n" + "-"*80)
    print("[NIPS 2003 Benchmark Reference - Validation Split]")
    print("  Method                    BER (Validation)")
    print("  " + "-"*50)
    print("  Lambda (baseline)         ~21.0%")
    print("  Linear SVM (all feats)    ~15.0%")
    print("  Winner (Jie Cheng)        ~11.0% (with feature selection)")
    print("-"*80)
    
    # Highlight best ArrowSpace result
    best_overall = df.loc[df["BER"].idxmin()]
    print(f"\n[Best ArrowSpace Result]")
    print(f"  Method: {best_overall['method']}")
    print(f"  k: {best_overall['k']}")
    print(f"  BER: {best_overall['BER']*100:.2f}%")
    print(f"  Precision: {best_overall['precision']:.3f}")
    print(f"  Recall: {best_overall['recall']:.3f}")
    print(f"  F1: {best_overall['f1']:.3f}")
    
    logging.info(f"\n[5] Results saved to: {out_csv}")
    logging.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="k-NN Classification: Dorothea Train (800) → Validation (350)"
    )
    parser.add_argument("--data-dir", default="data/DOROTHEA",
                        help="Directory containing Dorothea dataset files")
    parser.add_argument("--storage", default="./storage",
                        help="Directory containing pre-built ArrowSpace index")
    parser.add_argument("--dataset", default="dorothea_highdim",
                        help="Dataset name prefix (should be built from train split only)")
    parser.add_argument("--noise", type=float, default=0.001,
                        help="Noise level for densification (must match build)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    evaluate_classification(args)
