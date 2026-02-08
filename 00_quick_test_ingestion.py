#!/usr/bin/env python3
"""
Ingest Dorothea (UCI / NIPS 2003) sparse-binary dataset and build an ArrowSpace 
Laplacian graph using the internal dimensionality reduction harness.

Builds the space on Train + Valid and executes search tests on the Test set.

python ingestion.py --data-dir data/DOROTHEA/
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from arrowspace import ArrowSpaceBuilder, set_debug

import logging
logging.basicConfig(level=logging.INFO)

def read_sparse_binary_indices(path: Path, n_features: int) -> sp.csr_matrix:
    """Read Dorothea 1-based indices into a CSR matrix."""
    indptr = [0]
    indices = []
    data = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                indptr.append(indptr[-1])
                continue
            toks = line.split()
            row_idx = [int(t) - 1 for t in toks if t]
            row_idx = [j for j in row_idx if 0 <= j < n_features]
            row_idx.sort()

            indices.extend(row_idx)
            data.extend([1.0] * len(row_idx))
            indptr.append(len(indices))

    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(indptr) - 1, n_features)
    )

def gl_to_scipy_csr(gl):
    """Convert ArrowSpace GraphLaplacian to SciPy CSR."""
    data_f32, indices_u64, indptr_u64, shape = gl.tocsr()
    return sp.csr_matrix(
        (np.asarray(data_f32, dtype=np.float64),
         np.asarray(indices_u64, dtype=np.int32),
         np.asarray(indptr_u64, dtype=np.int64)),
        shape=tuple(shape)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to Dorothea files")
    ap.add_argument("--n-features", type=int, default=100000)
    
    # ArrowSpace graph params
    ap.add_argument("--eps", type=float, default=0.970636)
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.7, help="Spectral vs Semantic balance")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(data_dir)
    set_debug(args.debug)

    # 1) Load Indexing Data: Train + Valid [file:4]
    print("Ingesting Train + Valid for building space...")
    X_train = read_sparse_binary_indices(data_dir / "dorothea_train.data", args.n_features)
    X_valid = read_sparse_binary_indices(data_dir / "dorothea_valid.data", args.n_features)
    # X_build = sp.vstack([X_train, X_valid]).toarray().astype(np.float64)

    # 2) Build ArrowSpace (Internal JL harness handles 100k -> 1024 dims) [file:3]
    graphparams = {
        "eps": args.eps,
        "k": args.k,
        "topk": args.topk,
        "p": args.p,
    }
    if args.sigma >= 0.0:
        graphparams["sigma"] = args.sigma

    print(f"Building ArrowSpace on {X_train.shape} matrix...")
    start = time.perf_counter()
    builder = (ArrowSpaceBuilder()
            .with_dims_reduction(enabled=True, eps=args.eps / 2.0)
            .with_sampling("simple", 1.0))
    aspace, gl = builder.build_and_store(graphparams, X_train.toarray().astype(np.float64))
    print(f"Build time: {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    main()
