#!/usr/bin/env python3
"""
Prepare Dorothea for high-dimensional ArrowSpace benchmarking.
Uses 'Spectral Diffusion' (Noise Injection) to densify the 100k feature space
without dimensionality reduction, showcasing ArrowSpace's bandwidth and 
Lambda-Tau ranking capabilities on massive vectors.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize

def load_and_transform_highdim(storage_dir: Path, target_dim: int = 0, noise_level: float = 0.001):
    # 1. Load Metadata & Data
    json_files = list(storage_dir.glob("*-raw_input_metadata.json"))
    if not json_files:
        raise FileNotFoundError("No metadata found.")
    
    with open(json_files[0]) as f:
        meta = json.load(f)
    
    parquet_file = storage_dir / meta['files']['matrix']['filename']
    print(f"Loading {parquet_file.name}...")
    
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    data_cols = [c for c in df.columns if c.startswith('col_')]
    X_raw = df[data_cols].values
    
    n_samples, n_features = X_raw.shape
    print(f"Original Data: {n_samples} items x {n_features} features")

    # 2. Strategy Selection
    if target_dim <= 0 or target_dim == n_features:
        print(f"\n[Strategy] Full-Dimension Densification (100k dims)")
        print("Injecting Gaussian noise to create a continuous manifold in original space...")
        
        # Create dense noise matrix (N x 100k)
        # Note: This requires ~900MB RAM for Dorothea (1150 x 100k x 8bytes)
        # Safe for most modern machines.
        noise = np.random.normal(0, noise_level, size=(n_samples, n_features))
        
        # Add noise to original sparse signal
        X_transformed = X_raw + noise
        
    else:
        print(f"\n[Strategy] High-Dimensional Projection (Target: {target_dim} dims)")
        print("Projecting sparse signal to high-dimensional dense subspace...")
        
        # Use Gaussian Projection to a large target (e.g., 16384)
        rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
        X_transformed = rp.fit_transform(X_raw)

    # 3. Normalization (Critical for High-Dim Geometry)
    # L2 normalization ensures all vectors lie on the high-dimensional hypersphere
    X_final = normalize(X_transformed, norm='l2', axis=1)
    
    print(f"Final Shape: {X_final.shape}")
    print(f"Final Size:  {X_final.nbytes / 1024**2:.2f} MB")
    
    # 4. Preview
    print("\nSample Vector (First 10 dims):")
    print(np.round(X_final[0, :10], 5))
    
    # 5. Save
    # Save as .npy because Parquet overhead on wide dense tables can be high
    dim_tag = "full100k" if X_final.shape[1] == 100000 else f"dim{X_final.shape[1]}"
    out_file = storage_dir / f"dorothea_highdim_{dim_tag}.npy"
    np.save(out_file, X_final)
    print(f"\nSaved high-dimensional data to {out_file}")
    print("Use ArrowSpaceBuilder.buildfull() with this file to verify throughput.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_dir", type=Path, nargs="?", default="./storage")
    parser.add_argument("--dim", type=int, default=0, 
                        help="Target dimension. 0 = keep original 100k (MAX stress test), 16384 = high projection.")
    args = parser.parse_args()
    
    if args.storage_dir.exists():
        load_and_transform_highdim(args.storage_dir, args.dim)
    else:
        print("Invalid storage directory.")
