#!/usr/bin/env python3
"""
Prepare Dorothea for high-dimensional ArrowSpace benchmarking.

Uses seeded 'Spectral Diffusion' (Gaussian noise injection) to densify the feature space
and L2-normalizes rows, producing a reproducible dense manifold for ArrowSpace.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize


def load_and_transform_highdim(
    storage_dir: Path,
    target_dim: int = 0,
    noise_level: float = 0.001,
    seed: int = 42,
    dtype=np.float64,
):
    # 1) Load metadata & raw input
    json_files = list(storage_dir.glob("*-raw_input_metadata.json"))
    if not json_files:
        raise FileNotFoundError("No metadata found.")

    with open(json_files[0], "r", encoding="utf-8") as f:
        meta = json.load(f)

    parquet_file = storage_dir / meta["files"]["matrix"]["filename"]
    print(f"Loading {parquet_file.name}...")

    df = pd.read_parquet(parquet_file, engine="pyarrow")
    data_cols = [c for c in df.columns if c.startswith("col_")]
    X_raw = df[data_cols].values.astype(dtype, copy=False)

    n_samples, n_features = X_raw.shape
    print(f"Original Data: {n_samples} items x {n_features} features")

    rng = np.random.default_rng(seed)

    # 2) Strategy selection
    if target_dim <= 0 or target_dim == n_features:
        print("\n[Strategy] Full-Dimension Densification (100k dims)")
        print(f"Injecting Gaussian noise with seed={seed}, noise_level={noise_level}...")

        # Deterministic noise
        noise = rng.normal(0.0, noise_level, size=(n_samples, n_features)).astype(dtype, copy=False)
        X_transformed = X_raw + noise

    else:
        print(f"\n[Strategy] High-Dimensional Projection (Target: {target_dim} dims)")
        print("Projecting sparse signal to high-dimensional dense subspace...")

        # Keep projection deterministic too (separate from noise seed to avoid coupling)
        rp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
        X_proj = rp.fit_transform(X_raw).astype(dtype, copy=False)

        print(f"Injecting Gaussian noise in projected space with seed={seed}, noise_level={noise_level}...")
        noise = rng.normal(0.0, noise_level, size=X_proj.shape).astype(dtype, copy=False)
        X_transformed = X_proj + noise

    # 3) Normalization (critical for high-dim geometry)
    X_final = normalize(X_transformed, norm="l2", axis=1)

    print(f"Final Shape: {X_final.shape}")
    print(f"Final Size:  {X_final.nbytes / 1024**2:.2f} MB")

    # 4) Preview
    print("\nSample Vector (First 10 dims):")
    print(np.round(X_final[0, :10], 5))

    # 5) Save (.npy)
    dim_tag = "full100k" if X_final.shape[1] == 100000 else f"dim{X_final.shape[1]}"
    out_file = storage_dir / f"dorothea_highdim_{dim_tag}.npy"
    np.save(out_file, X_final)

    # Save preprocessing provenance for reproducibility
    prep_meta = {
        "source_parquet": parquet_file.name,
        "n_samples": int(n_samples),
        "n_features_raw": int(n_features),
        "target_dim": int(target_dim),
        "noise_level": float(noise_level),
        "seed": int(seed),
        "dtype": str(np.dtype(dtype)),
        "output_file": out_file.name,
    }
    meta_out = storage_dir / f"dorothea_highdim_{dim_tag}_prep.json"
    meta_out.write_text(json.dumps(prep_meta, indent=2), encoding="utf-8")

    print(f"\nSaved high-dimensional data to {out_file}")
    print(f"Saved preprocessing metadata to {meta_out}")
    print("Use ArrowSpaceBuilder().build_full() with this file to verify throughput.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_dir", type=Path, nargs="?", default=Path("./storage"))
    parser.add_argument(
        "--dim",
        type=int,
        default=0,
        help="Target dimension. 0 = keep original 100k (MAX stress test), 16384 = high projection.",
    )
    parser.add_argument("--noise", type=float, default=0.001, help="Gaussian noise stddev for densification.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic noise (and projection).")
    args = parser.parse_args()

    if args.storage_dir.exists():
        load_and_transform_highdim(args.storage_dir, target_dim=args.dim, noise_level=args.noise, seed=args.seed)
    else:
        print("Invalid storage directory.")
