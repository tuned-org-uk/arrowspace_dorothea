#!/usr/bin/env python3
"""
Estimate ArrowSpace parameters for high-dimensional Dorothea 
and export as a Rust-compatible JSON configuration.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist

import logging
logging.basicConfig(level=logging.INFO)

def estimate_graph_params(X: np.ndarray):
    """
    Analyzes high-dim geometry to estimate wiring parameters.
    Targets the 15th percentile of distances for the manifold radius. [file:1]
    """
    n_items, n_dims = X.shape
    print(f"Estimating parameters for {n_items} items in {n_dims} dimensions...")
    
    # Sample distances for heuristic
    sample_size = min(n_items, 300)
    sample_idx = np.random.choice(n_items, sample_size, replace=False)
    dists = pdist(X[sample_idx], metric='cosine')
    
    # Heuristics based on distance concentration findings
    est_eps = np.percentile(dists, 15)
    est_k = int(np.ceil(2 * np.log2(n_items)))
    est_sigma = np.clip(np.std(dists) * 2, 0.1, 1.5)
    
    # Rust-compatible structure for GraphParams [file:1][file:3]
    params = {
        "eps": float(est_eps),
        "k": int(est_k),
        "topk": int(est_k // 2),
        "p": 2.0,
        "sigma": float(est_sigma)  # Exported as f64; Rust reads into Option<f64>
    }
    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-path", required=True, help="Path to high-dim .npy matrix")
    ap.add_argument("--out-json", default="./storage/estimated_graph_params.json", help="Output JSON path")
    args = ap.parse_args()

    # 1. Load Matrix
    X_build = np.load(args.npy_path)
    
    # 2. Estimate
    graph_params = estimate_graph_params(X_build)
    
    # 3. Export to JSON
    # This file can be read directly in Rust using serde_json
    with open(args.out_json, "w") as f:
        json.dump(graph_params, f, indent=2)
    
    print("-" * 30)
    print(f"Estimation Complete.")
    print(f"JSON Exported to: {args.out_json}")
    print(f"Parameters: {json.dumps(graph_params)}")
    print("-" * 30)

if __name__ == "__main__":
    main()
