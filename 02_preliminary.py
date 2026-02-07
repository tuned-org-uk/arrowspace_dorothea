#!/usr/bin/env python3
"""
Analyze high-dimensional Dorothea dense matrix (100k dims).
Reads the .npy dense matrix and uses the metadata JSON to correlate stats.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

import logging
logging.basicConfig(level=logging.INFO)

def load_metadata(json_path: Path) -> Dict[str, Any]:
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_highdim_dense(storage_dir: Path):
    """
    Loads high-dimensional dense matrix (.npy) and analyzes spectral characteristics.
    """
    # 1. Load Metadata
    json_files = list(storage_dir.glob("*-raw_input_metadata.json"))
    if not json_files:
        raise FileNotFoundError(f"No metadata found in {storage_dir}")
    
    metadata = load_metadata(json_files[0])
    dataset_name = metadata['name_id']
    
    # 2. Find and Load Dense Matrix
    # Prioritizes the transformed high-dim npy file
    npy_files = list(storage_dir.glob("dorothea_highdim_*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy dense matrix found in {storage_dir}")
    
    npy_path = npy_files[0]
    print(f"Loading high-dim dense matrix: {npy_path.name}")
    X = np.load(npy_path)
    
    n_items, n_features = X.shape
    print("-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Shape:   {n_items} items x {n_features} features")
    print(f"Memory:  {X.nbytes / 1024**2:.2f} MB")
    print("-" * 60)

    # 3. High-Dimensional Statistical Analysis
    # In 100k dims, sparsity should be ~0% due to noise injection
    sparsity = (X == 0).sum() / X.size
    row_norms = np.linalg.norm(X, axis=1)
    
    print("\n[Spectral Metrics]")
    print(f"  Global Sparsity:        {sparsity:.2%}")
    print(f"  Mean Row Norm (L2):     {row_norms.mean():.4f}")
    print(f"  Norm Std Dev:           {row_norms.std():.4f}")
    
    # Value distribution stats
    print(f"\n[Value Range]")
    print(f"  Min:  {X.min():.6f}")
    print(f"  Max:  {X.max():.6f}")
    print(f"  Mean: {X.mean():.6e}")

    # 4. Visualization (High-Dim Context)
    print("\nGenerating high-dimensional distribution plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Value Distribution (Log scale to see noise floor vs signal)
    sns.histplot(X.flatten()[::1000], bins=100, ax=ax1, kde=True, color="teal", log_scale=(False, True))
    ax1.set_title("Feature Value Distribution (Sampled, Log Scale)")
    ax1.set_xlabel("Value (Original + Noise)")
    ax1.set_ylabel("Frequency")
    
    # Plot 2: Pairwise Distance Distribution (Sampled)
    # This showcases the "Distance Concentration" in high dimensions
    print("Computing sampled pairwise distances...")
    sample_indices = np.random.choice(n_items, min(n_items, 200), replace=False)
    X_sample = X[sample_indices]
    from scipy.spatial.distance import pdist
    dists = pdist(X_sample, metric='cosine')
    
    sns.histplot(dists, bins=40, ax=ax2, kde=True, color="indigo")
    ax2.set_title("Pairwise Cosine Distance Distribution")
    ax2.set_xlabel("Cosine Distance")
    ax2.set_ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(storage_dir / "highdim_spectral_analysis.png", dpi=150)
    print(f"Saved analysis to {storage_dir / 'highdim_spectral_analysis.png'}")
    plt.show()

if __name__ == "__main__":
    import sys
    # Use same logic as previous module for path resolution [file:24]
    storage_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./storage")
    if not storage_dir.exists():
        storage_dir = Path("./dorothea_out/storage")
        
    if storage_dir.exists():
        analyze_highdim_dense(storage_dir)
    else:
        print("Storage directory not found.")
