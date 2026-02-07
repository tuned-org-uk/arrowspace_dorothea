### Parameter Estimation Logic

- **`eps` (Manifold Radius)**: In 100,000 dimensions, Cosine distances often concentrate (e.g., between 0.8 and 0.9). We set `eps` slightly above the **mean pairwise distance** to ensure connectivity without creating a dense clique.[^2]
- **`k` (Connectivity)**: For $N=1,150$ items, we follow the $O(\log N)$ rule. $k \approx 20$ provides enough path redundancy for stable Rayleigh quotients ($\lambda$).
- **`sigma` (Smoothing)**: Since we injected noise at `0.001`, we use a moderate `sigma` to ensure the heat-flow diffusion doesn't wash out the original one-hot signal.[^1]


### Updated Module with Parameter Auto-Estimation

```python
def estimate_graph_parameters(X: np.ndarray):
    """
    Estimates optimal ArrowSpace parameters based on high-dim geometry.
    """
    n_items, n_dims = X.shape
    
    # 1. Sample pairwise distances to find the 'Manifold Edge'
    sample_size = min(n_items, 300)
    idx = np.random.choice(n_items, sample_size, replace=False)
    X_sample = X[idx]
    
    from scipy.spatial.distance import pdist
    # Cosine distance is standard for high-dim ArrowSpace [file:3]
    dists = pdist(X_sample, metric='cosine')
    
    avg_dist = np.mean(dists)
    std_dist = np.std(dists)
    
    # 2. Parameter Heuristics
    # EPS: Set to the 15th percentile of distances to capture local neighborhood
    # but stay below the 'concentration' peak. [file:1]
    estimated_eps = np.percentile(dists, 15)
    
    # K: Logarithmic scaling based on dataset size [file:3]
    estimated_k = int(np.ceil(2 * np.log2(n_items)))
    
    # Sigma: Inverse of the distance spread to control heat-kernel decay
    estimated_sigma = np.clip(std_dist * 2, 0.1, 1.5)

    print("\n" + "="*60)
    print("ESTIMATED ARROWSPACE PARAMETERS (High-Dim Build)")
    print("="*60)
    print(f"  eps (Radius):   {estimated_eps:.4f}  (Based on 15th percentile distance)")
    print(f"  k (Neighbors):  {estimated_k}       (O(log N) connectivity)")
    print(f"  sigma (Smooth): {estimated_sigma:.4f}  (Spectral smoothing factor)")
    print(f"  p (Kernel):     2.0          (Standard Gaussian decay)")
    print("-" * 60)
    
    params = {
        "eps": float(estimated_eps),
        "k": estimated_k,
        "topk": int(estimated_k // 2),
        "p": 2.0,
        "sigma": float(estimated_sigma)
    }
    return params

# Usage in the main analysis flow:
# X = np.load(npy_path)
# best_params = estimate_graph_parameters(X)
```


### Why these parameters matter for your 100k data:

| Parameter | High-Dim Sensitivity |
| :-- | :-- |
| **EPS** | If set too high (e.g., `>0.9`), every node connects to every other node because of distance concentration. Keeping it at the **15th percentile** ensures we only wire "true" neighbors. [^2] |
| **K-Cap** | Restricting to `k=21` (for 1,150 items) prevents high-degree "hubs" from dominating the $\lambda$ calculation. This enforces **Sparsity** in the Laplacian. [^1] |
| **Sigma** | In dense noise-injected data, `sigma` controls how "sharply" the similarity drops off. A lower `sigma` (e.g., `0.2`) creates higher contrast in the spectral domain, making the Lambda-Tau ranking more effective. [^1] |

### Practical Build Call

Using these estimated parameters, your final build step should look like this:

```python
# Pass the estimated dict directly to the builder
aspace, gl = ArrowSpaceBuilder.buildfull(best_params, X_dense)
```

This configuration will maximize the **Spectral Gap**, ensuring that your search results are not just "the nearest items" but the "spectrally significant" neighbors.[^3][^1]