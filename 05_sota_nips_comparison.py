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
import seaborn as sns
from collections import Counter

from arrowspace import ArrowSpaceBuilder

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
    Returns: +1 or -1 prediction.
    """
    results = aspace.search(query, gl, tau=tau)[:k]
    neighbor_labels = [train_labels[idx] for idx, _ in results]
    return int(np.sign(np.sum(neighbor_labels)))


def knn_predict_cosine_baseline(X_train, y_train, query, k: int) -> int:
    """
    Pure cosine k-NN (no ArrowSpace) for baseline comparison.
    """
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
    Balanced Error Rate: average of error rates on positive and negative classes.
    This is the official Dorothea benchmark metric (NIPS 2003).
    
    BER = 0.5 * (FP/(FP+TN) + FN/(FN+TP))
    
    Returns dict with BER, precision, recall, and confusion matrix.
    """
    # Positive class: +1, Negative class: -1
    pos_mask = (y_true == 1)
    neg_mask = (y_true == -1)
    
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    # True Positives / False Negatives
    tp = ((y_pred == 1) & pos_mask).sum()
    fn = ((y_pred == -1) & pos_mask).sum()
    
    # True Negatives / False Positives
    tn = ((y_pred == -1) & neg_mask).sum()
    fp = ((y_pred == 1) & neg_mask).sum()
    
    # Error rates per class
    pos_error = fn / n_pos if n_pos > 0 else 0.0
    neg_error = fp / n_neg if n_neg > 0 else 0.0
    
    ber = 0.5 * (pos_error + neg_error)
    
    # Additional metrics
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
        "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }


# ============================================================================
# 4. EVALUATION PIPELINE
# ============================================================================

def evaluate_classification(args):
    """
    Main evaluation: k-NN classification with multiple tau and k settings.
    """
    data_dir = Path(args.data_dir)
    storage = Path(args.storage)
    
    print("="*70)
    print("ArrowSpace k-NN Classification Evaluation on Dorothea")
    print("="*70)
    
    # A. Load Training Data
    print("\n[1] Loading training data...")
    X_train_raw = read_sparse_binary(data_dir / "dorothea_train.data", n_features=100000)
    y_train = read_labels(data_dir / "dorothea_train.labels")
    
    print(f"   Training samples: {len(X_train_raw)}")
    print(f"   Positive: {(y_train == 1).sum()}, Negative: {(y_train == -1).sum()}")
    
    # B. Densify Training Set
    print("\n[2] Densifying training data (noise + L2 norm)...")
    X_train = densify_seeded(X_train_raw, noise_level=args.noise, seed=args.seed)
    
    # C. Build ArrowSpace
    print("\n[3] Building ArrowSpace index on training set...")
    graph_params = {
        "eps": args.eps,
        "k": args.graph_k,
        "topk": args.topk,
        "p": args.p,
        "sigma": args.sigma,
    }
    print(f"   Graph params: {graph_params}")
    
    import time
    start = time.time()
    aspace, gl = ArrowSpaceBuilder().build_full(graph_params, X_train)
    build_time = time.time() - start
    print(f"   Build time: {build_time:.2f}s")
    
    # D. Load Test Data
    print("\n[4] Loading test data...")
    
    # Try to load official test split (may be withheld)
    test_data_path = data_dir / "dorothea_test.data"
    test_labels_path = data_dir / "dorothea_test.labels"
    
    if test_data_path.exists() and test_labels_path.exists():
        print("   Using official test split")
        X_test_raw = read_sparse_binary(test_data_path, n_features=100000)
        y_test = read_labels(test_labels_path)
    else:
        # Fall back to validation set
        print("   Official test labels not found, using validation split")
        X_test_raw = read_sparse_binary(data_dir / "dorothea_valid.data", n_features=100000)
        y_test = read_labels(data_dir / "dorothea_valid.labels")
    
    X_test = densify_seeded(X_test_raw, noise_level=args.noise, seed=args.seed)
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Positive: {(y_test == 1).sum()}, Negative: {(y_test == -1).sum()}")
    
    # E. Run Classification Experiments
    print("\n[5] Running k-NN classification experiments...")
    
    tau_configs = {
        "Cosine (τ=1.0)": 1.0,
        "Hybrid (τ=0.72)": 0.72,
        "TauMode (τ=0.42)": 0.42,
    }
    
    k_values = args.k_values
    
    results = []
    
    for method_name, tau in tau_configs.items():
        for k in k_values:
            print(f"   Evaluating {method_name}, k={k}...")
            
            predictions = []
            for q in tqdm(X_test, desc=f"  {method_name} k={k}", leave=False):
                pred = knn_predict(aspace, gl, q, k=k, tau=tau, train_labels=y_train)
                predictions.append(pred)
            
            metrics = compute_ber(y_test, np.array(predictions))
            
            results.append({
                "method": method_name,
                "tau": tau,
                "k": k,
                **metrics,
            })
    
    # F. Baseline: Pure Cosine k-NN (no ArrowSpace)
    print("\n[6] Running baseline: Pure Cosine k-NN (no spectral graph)...")
    
    for k in k_values:
        print(f"   Baseline k={k}...")
        predictions = []
        for q in tqdm(X_test, desc=f"  Baseline k={k}", leave=False):
            pred = knn_predict_cosine_baseline(X_train, y_train, q, k=k)
            predictions.append(pred)
        
        metrics = compute_ber(y_test, np.array(predictions))
        
        results.append({
            "method": "Baseline Cosine",
            "tau": 1.0,  # equivalent to cosine-only
            "k": k,
            **metrics,
        })
    
    # G. Save Results
    df = pd.DataFrame(results)
    df = df.sort_values(["method", "k"])
    
    out_csv = storage / "dorothea_classification_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[7] Results saved to {out_csv}")
    
    # H. Summary Report
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*70)
    
    # Find best result per method
    best_per_method = df.loc[df.groupby("method")["BER"].idxmin()]
    
    print("\nBest BER per method:")
    print(best_per_method[["method", "k", "BER", "precision", "recall", "f1"]].to_string(index=False))
    
    # Reference benchmarks
    print("\n" + "-"*70)
    print("NIPS 2003 Benchmark Reference (from literature):")
    print("  Linear SVM (all features):  ~15.0% BER")
    print("  Lambda method (baseline):   ~21.0% BER")
    print("  Winner (Jie Cheng, 2003):   ~11.0% BER (with feature selection)")
    print("-"*70)
    
    # I. Visualization
    print("\n[8] Generating visualizations...")
    
    # Plot 1: BER vs k for each method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for method in df["method"].unique():
        subset = df[df["method"] == method]
        ax1.plot(subset["k"], subset["BER"] * 100, marker="o", label=method, linewidth=2)
    
    ax1.set_xlabel("k (number of neighbors)", fontsize=11)
    ax1.set_ylabel("Balanced Error Rate (%)", fontsize=11)
    ax1.set_title("k-NN Classification: BER vs k", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(y=15.0, color="red", linestyle="--", alpha=0.5, label="SVM baseline (~15%)")
    ax1.axhline(y=11.0, color="green", linestyle="--", alpha=0.5, label="SOTA 2003 (~11%)")
    
    # Plot 2: Precision-Recall trade-off
    for method in df["method"].unique():
        subset = df[df["method"] == method]
        ax2.scatter(subset["recall"], subset["precision"], label=method, s=80, alpha=0.7)
    
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Trade-off", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    out_png = storage / "dorothea_classification_plots.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"   Saved plots to {out_png}")
    
    # J. Detailed Report (Best Config Analysis)
    print("\n[9] Detailed Analysis of Best Configuration...")
    
    best_overall = df.loc[df["BER"].idxmin()]
    print(f"\nBest Overall Result:")
    print(f"  Method: {best_overall['method']}")
    print(f"  k: {best_overall['k']}")
    print(f"  BER: {best_overall['BER']*100:.2f}%")
    print(f"  Precision: {best_overall['precision']:.3f}")
    print(f"  Recall: {best_overall['recall']:.3f}")
    print(f"  F1: {best_overall['f1']:.3f}")
    print(f"  Confusion Matrix: {best_overall['confusion']}")
    
    # K. Statistical Summary
    summary = {
        "dataset": "Dorothea (NIPS 2003)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": 100000,
        "build_time_sec": float(build_time),
        "graph_params": graph_params,
        "best_method": best_overall["method"],
        "best_k": int(best_overall["k"]),
        "best_ber": float(best_overall["BER"]),
        "improvement_vs_baseline": None,
    }
    
    # Compare against baseline
    baseline_best = df[df["method"] == "Baseline Cosine"]["BER"].min()
    if baseline_best > 0:
        improvement = (baseline_best - best_overall["BER"]) / baseline_best * 100
        summary["improvement_vs_baseline"] = f"{improvement:.1f}%"
        print(f"\nImprovement over Baseline Cosine: {improvement:.1f}%")
    
    out_summary = storage / "dorothea_classification_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[10] Summary saved to {out_summary}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ArrowSpace k-NN Classification Evaluation on Dorothea"
    )
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="data/DOROTHEA",
                        help="Directory containing dorothea_{train,test}.{data,labels}")
    parser.add_argument("--storage", type=str, default="storage",
                        help="Output directory for results")
    
    # Densification
    parser.add_argument("--noise", type=float, default=0.001,
                        help="Noise level for densification (must match build)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducible densification")
    
    # ArrowSpace graph params (should match build)
    parser.add_argument("--eps", type=float, default=0.97)
    parser.add_argument("--graph-k", type=int, default=21)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    
    # Classification params
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 15, 20, 25],
                        help="List of k values for k-NN classification")
    
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    evaluate_classification(args)
