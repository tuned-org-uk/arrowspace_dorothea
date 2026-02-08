#!/usr/bin/env python3
"""
ArrowSpace k-NN Classification Evaluation on Dorothea (NIPS 2003).

Goal (no leakage):
- ArrowSpace index MUST be built from dorothea_train.data only (800 items).
- Evaluate classification on dorothea_valid.{data,labels} (350 queries, held-out).
- Report BER (Balanced Error Rate), plus precision/recall/F1.

Adds Script-04-style harness:
- Wraps aspace.search(...) in try/except
- Logs rich diagnostics per failed query (e.g., lambda==0 / "in the void")
- Writes failures to <dataset>_classification_zeroed.csv in storage/
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from arrowspace import load_arrowspace


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
# 2. METRICS (BER)
# ============================================================================

def compute_ber(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Balanced Error Rate: average of error rates on positive and negative classes.

    BER = 0.5 * (FP/(FP+TN) + FN/(FN+TP))
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == -1)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    tp = int(((y_pred == 1) & pos_mask).sum())
    fn = int(((y_pred == -1) & pos_mask).sum())
    tn = int(((y_pred == -1) & neg_mask).sum())
    fp = int(((y_pred == 1) & neg_mask).sum())

    pos_error = (fn / n_pos) if n_pos > 0 else 0.0
    neg_error = (fp / n_neg) if n_neg > 0 else 0.0
    ber = 0.5 * (pos_error + neg_error)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "BER": float(ber),
        "pos_error": float(pos_error),
        "neg_error": float(neg_error),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ============================================================================
# 3. SEARCH HARNESS + PREDICTORS
# ============================================================================

def _vector_stats(q: np.ndarray) -> dict:
    q = q.astype(np.float64, copy=False)
    return {
        "query_norm": float(np.linalg.norm(q)),
        "query_sparsity": float(np.mean(q == 0.0)),
        "query_mean": float(np.mean(q)),
        "query_std": float(np.std(q)),
        "query_min": float(np.min(q)),
        "query_max": float(np.max(q)),
        "query_dim": int(q.shape[0]),
    }


def safe_search(aspace, gl, q: np.ndarray, tau: float, query_idx: int, method_name: str):
    """
    Script-04-style harness around aspace.search.

    Returns:
      (results, None) on success
      (None, error_record) on failure
    """
    try:
        r = aspace.search(q, gl, tau=tau)
        return r, None

    except Exception as e:
        err = {
            "query_idx": int(query_idx),
            "method": method_name,
            "tau": float(tau),
            "error_type": type(e).__name__,
            "error_message": str(e),
            **_vector_stats(q),
            # Storing the full vector can be large; keep it optional via CLI flag.
        }
        return None, err


def knn_predict_from_results(results, k: int, train_labels: np.ndarray) -> int:
    """
    Majority-vote k-NN classification given ArrowSpace results [(idx, score), ...].
    """
    if results is None or len(results) == 0:
        return -1

    topk = results[:k]
    neighbor_labels = [train_labels[idx] for idx, _ in topk]
    if len(neighbor_labels) == 0:
        return -1

    s = int(np.sign(np.sum(neighbor_labels)))
    return s if s != 0 else -1  # tie-break to negative


def knn_predict_cosine_baseline(X_train: np.ndarray, y_train: np.ndarray, q: np.ndarray, k: int) -> int:
    """
    Pure cosine k-NN baseline (no ArrowSpace). Uses full dense X_train (800 x 100000).
    """
    from scipy.spatial.distance import cdist

    dists = cdist(q.reshape(1, -1), X_train, metric="cosine")[0]
    topk_idx = np.argpartition(dists, k)[:k]
    neighbor_labels = y_train[topk_idx]

    s = int(np.sign(np.sum(neighbor_labels)))
    return s if s != 0 else -1


# ============================================================================
# 4. EVALUATION
# ============================================================================

def evaluate_classification(args):
    data_dir = Path(args.data_dir)
    storage_dir = Path(args.storage).resolve()

    # ----------------------------------------------------------------------
    # A. Restore ArrowSpace index (must be train-only)
    # ----------------------------------------------------------------------
    graph_params = {"eps": args.eps, "k": args.graph_k, "topk": args.topk, "p": args.p, "sigma": args.sigma}

    logging.info(f"Loading ArrowSpace from storage: dataset={args.dataset} path={storage_dir}")
    aspace, gl = load_arrowspace(
        storage_path=str(storage_dir),
        dataset_name=args.dataset,
        graph_params=graph_params,
        energy=False,
    )
    logging.info(f"Loaded ArrowSpace: {aspace.nitems} items × {aspace.nfeatures} features")

    # Hard fail if the index contains anything other than train split.
    EXPECTED_TRAIN_N = 800
    if aspace.nitems != EXPECTED_TRAIN_N:
        raise ValueError(
            f"Data leakage risk: expected index size {EXPECTED_TRAIN_N} (train-only), got {aspace.nitems}. "
            f"Rebuild index from dorothea_train.data only and re-run."
        )

    # ----------------------------------------------------------------------
    # B. Load training labels (for voting) and training dense matrix (for baseline)
    # ----------------------------------------------------------------------
    y_train = read_labels(data_dir / "dorothea_train.labels")
    if len(y_train) != EXPECTED_TRAIN_N:
        raise ValueError(f"Expected 800 training labels, got {len(y_train)}")

    # Baseline needs X_train. (ArrowSpace itself doesn't need it.)
    X_train_raw = read_sparse_binary(data_dir / "dorothea_train.data", aspace.nfeatures)
    X_train = densify_seeded(X_train_raw, noise_level=args.noise, seed=args.seed)

    # ----------------------------------------------------------------------
    # C. Validation split as test set (test labels withheld)
    # ----------------------------------------------------------------------
    X_test_raw = read_sparse_binary(data_dir / "dorothea_valid.data", aspace.nfeatures)
    y_test = read_labels(data_dir / "dorothea_valid.labels")
    X_test = densify_seeded(X_test_raw, noise_level=args.noise, seed=args.seed)

    if args.n_queries is not None:
        X_test = X_test[: args.n_queries]
        y_test = y_test[: args.n_queries]

    logging.info(f"Evaluation queries: {len(X_test)} (validation split)")

    # ----------------------------------------------------------------------
    # D. Run experiments (with Script-04-style harness)
    # ----------------------------------------------------------------------
    tau_configs = {
        "Cosine": 1.0,
        "Hybrid": 0.72,
        "TauMode": 0.42,
    }

    k_values = args.k_values
    results_rows = []
    zeroed_rows = []  # per-query failures
    completed = {name: 0 for name in tau_configs.keys()}
    failed = {name: 0 for name in tau_configs.keys()}

    for method_name, tau in tau_configs.items():
        for k in k_values:
            logging.info(f"Evaluating method={method_name} tau={tau} k={k}")

            preds = []
            for i, q in enumerate(tqdm(X_test, desc=f"{method_name} k={k}", leave=False)):
                r, err = safe_search(aspace, gl, q, tau=tau, query_idx=i, method_name=method_name)

                if err is not None:
                    failed[method_name] += 1

                    if args.save_vectors:
                        err["vector"] = q.tolist()

                    # Print a Script-04-style block once per failure occurrence.
                    print(f"\n{'='*70}")
                    print(f"⚠️  QUERY FAILURE: Test query {i} | method={method_name} tau={tau} k={k}")
                    print(f"Error: {err['error_type']}: {err['error_message']}")
                    print(f"Stats: norm={err['query_norm']:.6f} mean={err['query_mean']:.3e} std={err['query_std']:.3e} "
                          f"min={err['query_min']:.3e} max={err['query_max']:.3e} dim={err['query_dim']}")
                    print(f"{'='*70}\n")

                    zeroed_rows.append(err)
                    preds.append(-1)  # safe fallback prediction
                    continue

                completed[method_name] += 1
                preds.append(knn_predict_from_results(r, k=k, train_labels=y_train))

            metrics = compute_ber(y_test, np.array(preds, dtype=np.int32))
            results_rows.append({
                "method": method_name,
                "tau": float(tau),
                "k": int(k),
                **metrics,
                "n_queries": int(len(X_test)),
                "n_failed_queries": int(failed[method_name]),
            })

    # Baseline cosine k-NN (no ArrowSpace) — no spectral failures expected.
    for k in k_values:
        logging.info(f"Evaluating baseline cosine k-NN, k={k}")
        preds = []
        for q in tqdm(X_test, desc=f"Baseline k={k}", leave=False):
            preds.append(knn_predict_cosine_baseline(X_train, y_train, q, k=k))
        metrics = compute_ber(y_test, np.array(preds, dtype=np.int32))
        results_rows.append({
            "method": "Baseline Cosine",
            "tau": 1.0,
            "k": int(k),
            **metrics,
            "n_queries": int(len(X_test)),
            "n_failed_queries": 0,
        })

    # ----------------------------------------------------------------------
    # E. Save outputs
    # ----------------------------------------------------------------------
    out_results = storage_dir / f"{args.dataset}_classification_results.csv"
    pd.DataFrame(results_rows).to_csv(out_results, index=False)
    logging.info(f"Saved classification results: {out_results}")

    if len(zeroed_rows) > 0:
        out_zeroed = storage_dir / f"{args.dataset}_classification_zeroed.csv"
        pd.DataFrame(zeroed_rows).to_csv(out_zeroed, index=False)
        logging.info(f"Saved query failure diagnostics: {out_zeroed}")

    # Console report
    df = pd.DataFrame(results_rows).sort_values(["method", "k"])
    print("\n[Best BER per method]")
    best = df.loc[df.groupby("method")["BER"].idxmin()]
    print(best[["method", "k", "BER", "precision", "recall", "f1", "n_failed_queries"]].to_string(index=False))

    print("\n[Failure counts by tau-mode]")
    for m in tau_configs.keys():
        print(f"  {m}: failed={failed[m]} completed={completed[m]} (total attempts={failed[m]+completed[m]})")


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ArrowSpace Dorothea k-NN (train-only index, valid evaluation)")

    p.add_argument("--data-dir", default="data/DOROTHEA", help="Dorothea dataset dir")
    p.add_argument("--storage", default="./storage", help="Storage dir containing ArrowSpace parquet/json")
    p.add_argument("--dataset", default="dorothea_highdim", help="Dataset prefix in storage/ (train-only index)")

    p.add_argument("--noise", type=float, default=0.001, help="Densification noise level")
    p.add_argument("--seed", type=int, default=42, help="Densification RNG seed")
    p.add_argument("--n-queries", type=int, default=50, help="How many validation queries to evaluate (None=all)")

    # Graph params (should match build)
    p.add_argument("--eps", type=float, default=0.97)
    p.add_argument("--graph-k", type=int, default=21)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=0.1)

    # Classification params
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 15, 25])

    # Diagnostics
    p.add_argument("--save-vectors", action="store_true", help="Include full query vector in *_zeroed.csv (large!)")

    args = p.parse_args()
    evaluate_classification(args)
