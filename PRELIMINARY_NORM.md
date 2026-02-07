In the preparation script (e.g., `prep_highdim_wiring.py`), we applied `normalize(X_transformed, norm='l2', axis=1)`. This forces every single vector in the 100k-dimensional space to lie exactly on the surface of a unit hypersphere. Because every vector now has a length of exactly $1.0$, there is **zero variance** in the magnitudes of the rows.

### Why this is actually good for Graph Wiring

This "spherical" data is ideal for **Cosine-based graph wiring** in ArrowSpace for several reasons:

* **Metric Stability**: In high dimensions (100k), magnitude differences in sparse vectors can be erratic. By normalizing to the unit sphere, you ensure that the only signal ArrowSpace's graph-builder considers is the **angular relationship** (correlation) between compounds.[^2]
* **Rayleigh Reliability**: ArrowSpace computes $\lambda$ (Rayleigh quotients) to find spectral roughness. When all nodes have unit norm, the $\lambda$ values strictly reflect the **topological connectivity** and neighborhood density, rather than just "loud" vectors having high scores.[^3][^2]
* **Comparison Fairness**: Since Dorothea is binary (one-hot), a compound with 10 active bits would naturally have a much larger norm than one with 2 bits. Normalization prevents the graph from being biased toward compounds that simply have more active features, allowing the spectral analysis to find structural similarities instead.[^4]


### How to interpret the stats

1. **Mean Row Norm = 1.0000**: Confirmation that the data is projected onto the unit hypersphere.
2. **Norm Std Dev = 0.0000**: Confirmation that the projection is perfectly uniform across all 1,150 items.
3. **Global Sparsity = 0.00%**: Confirmation that the noise injection successfully created a continuous manifold (no more "dead zones" in the graph).[^2]

### Next Step for the Graph parameters

When you run the parameter estimation, it will now focus entirely on the **Cosine Distance Distribution** (the right-hand plot in your analysis). Because the norms are identical, the `eps` parameter will purely define the "angular radius" of connectivity.

**Recommendation**: Proceed with the wiring using `eps` set to the **15th percentile** of these distances. This will wire the most correlated molecular structures together while ignoring the "flat" high-dimensional background.[^5]