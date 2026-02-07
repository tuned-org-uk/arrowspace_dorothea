```
/app/arrowspace_dorothea$ python 01_prepare_for_wiring.py 
Loading dataset_4d7dae-raw_input.parquet...
Original Data: 1150 items x 100000 features

[Strategy] Full-Dimension Densification (100k dims)
Injecting Gaussian noise to create a continuous manifold in original space...
Final Shape: (1150, 100000)
Final Size:  877.38 MB

Sample Vector (First 10 dims):
[-1.e-05 -4.e-05 -1.e-05  1.e-05  1.e-05  1.e-05  4.e-05 -8.e-05 -3.e-05
 -1.e-05]

Saved high-dimensional data to storage/dorothea_highdim_full100k.npy
Use ArrowSpaceBuilder().build_full() with this file to verify throughput.


/app/arrowspace_dorothea$ python 02_preliminary.py
Loading high-dim dense matrix: dorothea_highdim_full100k.npy
------------------------------------------------------------
Dataset: dataset_4d7dae-raw_input
Shape:   1150 items x 100000 features
Memory:  877.38 MB
------------------------------------------------------------

[Spectral Metrics]
  Global Sparsity:        0.00%
  Mean Row Norm (L2):     1.0000
  Norm Std Dev:           0.0000

[Value Range]
  Min:  -0.000203
  Max:  0.039179
  Mean: 2.973904e-04

Generating high-dimensional distribution plots...
Computing sampled pairwise distances...
Saved analysis to storage/highdim_spectral_analysis.png


/app/arrowspace_dorothea$ python 03_estimate_params_wiring.py --npy-path storage/dorothea_highdim_full100k.npy 
Estimating parameters for 1150 items in 100000 dimensions...
------------------------------
Estimation Complete.
JSON Exported to: ./storage/estimated_graph_params.json
Parameters: {"eps": 0.9703554777401838, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}
------------------------------


   Compiling dorothea_wiring v0.1.0 (/app/arrowspace_dorothea/dorothea_wiring)
    Finished `release` profile [optimized] target(s) in 1.24s
     Running `target/release/dorothea_wiring`
[2026-02-07T13:25:33Z INFO  dorothea_wiring] Starting Dorothea High-Dimensional Build Pipeline
[2026-02-07T13:25:34Z INFO  dorothea_wiring] Matrix loaded: 1150 items x 100000 features
[2026-02-07T13:25:34Z INFO  dorothea_wiring] Loaded spectral parameters: EstimatedParams { eps: 0.9703554777401838, k: 21, topk: 10, p: 2.0, sigma: Some(0.1) }
[2026-02-07T13:25:35Z INFO  dorothea_wiring] Wiring Laplacian Graph (Max-Stress Mode)...
[2026-02-07T13:25:35Z INFO  arrowspace::builder] Initializing new ArrowSpaceBuilder
[2026-02-07T13:25:35Z INFO  arrowspace::builder] Configuring lambda graph: eps=0.9703554777401838, k=21, p=2, sigma=Some(0.1)
[2026-02-07T13:25:35Z INFO  arrowspace::builder] Configuring inline sampling: None
[2026-02-07T13:25:35Z INFO  arrowspace::builder] Enabling persistence at: ./../storage
[2026-02-07T13:25:35Z INFO  arrowspace::builder] Building ArrowSpace from 1150 items with 100000 features
[2026-02-07T13:25:52Z INFO  arrowspace::builder] EigenMaps::start_clustering: N=1150 items, F=100000 features
[2026-02-07T13:25:53Z INFO  arrowspace::sampling] Simple random sampler with keep rate 100.0%
[2026-02-07T13:25:53Z INFO  arrowspace::builder] Computing optimal clustering parameters
[2026-02-07T13:25:53Z INFO  arrowspace::clustering] Computing optimal K for clustering: N=1150, F=100000
[2026-02-07T14:26:58Z INFO  arrowspace::builder] Running incremental clustering: max_clusters=11, radius=1.468942
[2026-02-07T14:26:58Z INFO  arrowspace::clustering] Starting incremental clustering with inline sampling
[2026-02-07T14:27:01Z INFO  arrowspace::builder] Clustering complete: 11 centroids, 1150 items assigned
[2026-02-07T14:27:01Z INFO  arrowspace::builder] Applying JL projection: 100000 features → 32 dimensions (ε=0.97)
[2026-02-07T14:27:02Z INFO  arrowspace::builder] Projection complete: 3125.0x compression, stored as 8-byte seed
[2026-02-07T14:27:02Z INFO  arrowspace::eigenmaps] EigenMaps::eigenmaps: Building Laplacian from 11 centroids × 32 features
[2026-02-07T14:27:02Z INFO  arrowspace::graph] Building Laplacian matrix for K cluster: 11 clusters
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Building Laplacian matrix for 11 items with 32 features
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Building CosinePair data structure
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Computing degrees for inline sparsification
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Computing k-NN with CosinePair: k=11
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Converting adjacency to sparse Laplacian matrix (DashMap batched)
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Sparse Laplacian construction time: 122.23µs
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Total Laplacian construction time: 1.434948ms
[2026-02-07T14:27:02Z INFO  arrowspace::laplacian] Successfully built sparse Laplacian matrix (11x11) with 288 non-zeros
[2026-02-07T14:27:02Z INFO  arrowspace::graph] Laplacian matrix built: 32×32 with 1150 nodes, 288 non-zeros
[2026-02-07T14:27:02Z INFO  arrowspace::eigenmaps] Laplacian construction complete: 32×32 matrix, 288 non-zeros, 71.88% sparse
[2026-02-07T14:27:02Z INFO  arrowspace::builder] Computing taumode lambdas with synthesis: Median
[2026-02-07T14:27:02Z INFO  arrowspace::eigenmaps] EigenMaps::compute_taumode: Computing λ values for 1150 items using Median
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ╔═════════════════════════════════════════════════════════════╗
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║          Parallel TauMode Lambda Computation                ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ╠═════════════════════════════════════════════════════════════╣
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║ Configuration:                                              ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Items:           1150                                     ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Features:        100000                                   ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Threads:         6                                        ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   TauMode:         Median                                   ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Graph Source:    Laplacian Matrix                         ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Graph Shape:     32×32                                   ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Graph NNZ:       288                                      ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ║   Graph Sparsity:  0.281250                                 ║
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] ╚═════════════════════════════════════════════════════════════╝
[2026-02-07T14:27:02Z INFO  arrowspace::taumode] Starting parallel lambda computation...
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ╔═════════════════════════════════════════════════════════════╗
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║          Computation Statistics                             ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ╠═════════════════════════════════════════════════════════════╣
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Sequential Items: 0                                       ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Parallel Items:   0                                       ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Compute Time:     6.038s                                  ║
[2026-02-07T14:27:08Z INFO  arrowspace::core] Updating lambdas with 1150 new values
[2026-02-07T14:27:08Z INFO  arrowspace::core] Normalized lambdas to [0, 1] range (original spread: 0.075219)
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Update Time:      36.049µs                                ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Total Time:       6.038s                                  ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ║   Throughput:       190                                     items/sec ║
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ╚═════════════════════════════════════════════════════════════╝
[2026-02-07T14:27:08Z INFO  arrowspace::taumode] ✓ Parallel taumode lambda computation completed successfully
[2026-02-07T14:27:08Z INFO  arrowspace::eigenmaps] λ computation complete: min=0.000000, max=1.000000, mean=0.120594
[2026-02-07T14:27:08Z INFO  arrowspace::builder] Total ArrowSpaceBuilder construction time: 3692.790685089s
[2026-02-07T14:27:08Z INFO  arrowspace::builder] ArrowSpace build completed successfully
[2026-02-07T14:27:08Z INFO  dorothea_wiring] ArrowSpace Build Success in 3692.85s
[2026-02-07T14:27:08Z INFO  dorothea_wiring] Final Graph Stats:
[2026-02-07T14:27:08Z INFO  dorothea_wiring]   - Nodes:     1150
[2026-02-07T14:27:08Z INFO  dorothea_wiring]   - Laplacian: (32, 32)
[2026-02-07T14:27:08Z INFO  dorothea_wiring]   - Items:     1150
```