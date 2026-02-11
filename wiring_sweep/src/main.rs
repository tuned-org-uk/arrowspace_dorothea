use arrowspace::builder::ArrowSpaceBuilder;
use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::path::PathBuf;
use std::io::BufRead;
use log::{warn, info};


pub fn run_grid_search(
    dataset: Vec<Vec<f64>>, 
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Better k sweep: fewer values, more strategic
    let k_sweep = vec![5, 10, 25, 50, 100];
    
    // JL epsilon (rpeps): controls projected dimension
    // Lower = more dims = slower but more accurate
    let jl_epsilons = vec![0.25, 0.35, 0.50, 0.70];
    
    // Cluster radius: squared L2 distance threshold
    // Higher = fewer, larger clusters
    let cluster_radii = vec![0.8, 1.2, 1.6, 2.0, 2.6, 3.2];
    
    // Lambda-graph eps: cosine distance threshold for edges
    // THIS IS THE KEY PARAMETER - controls graph sparsity
    // Higher = more edges = denser Laplacian = non-zero lambdas
    let lambda_eps_sweep = vec![0.05, 0.10, 0.20, 0.30, 0.50];
    
    // Lambda k: max neighbors per node in graph
    let lambda_k = 10; // Fixed for now, can sweep later
    
    let total = k_sweep.len() * jl_epsilons.len() * cluster_radii.len() * lambda_eps_sweep.len();
    
    // Create CSV file with all parameters
    let mut file = File::create(output_path)?;
    writeln!(
        file, 
        "k,jl_eps,cluster_radius,lambda_eps,lambda_k,jl_dim,actual_clusters,lambda_min,lambda_max,lambda_mean,lambda_spread,graph_nnz,graph_sparsity"
    )?;
    
    let mut count = 0;
    
    for &k in &k_sweep {
        for &jl_eps in &jl_epsilons {
            // Compute JL target dimension based on k (number of clusters)
            let jl_dim = ((8.0 * (k as f64).ln()) / (jl_eps * jl_eps)).ceil() as usize;
            let target_dim = jl_dim.min(2000).max(32); // Clamp to reasonable range
            
            for &cluster_radius in &cluster_radii {
                for &lambda_eps in &lambda_eps_sweep {
                    count += 1;
                    info!(
                        "Experiment {}/{}: k={}, jl_eps={:.2}, radius={:.1}, lambda_eps={:.2}", 
                        count, total, k, jl_eps, cluster_radius, lambda_eps
                    );
                    
                    let (aspace, gl) = ArrowSpaceBuilder::new()
                        .with_dims_reduction(true, Some(jl_eps))
                        .with_cluster_max_clusters(k)
                        .with_cluster_radius(cluster_radius)
                        // KEY FIX: sweep lambda_eps to control graph density
                        .with_lambda_graph(
                            lambda_eps,  // eps: cosine distance threshold
                            lambda_k,    // k: max neighbors
                            lambda_k/2,  // topk: results to return
                            2.0,         // p: kernel exponent
                            None         // sigma: use eps as default
                        )
                        .build(dataset.clone());
                    
                    // Extract metrics
                    let lambdas = aspace.lambdas();
                    let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let lambda_max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
                    let lambda_spread = lambda_max - lambda_min;
                    let nnz = gl.nnz();
                    let graph_shape = gl.shape();
                    let graph_sparsity = 1.0 - (nnz as f64 / (graph_shape.0 * graph_shape.1) as f64);
                    
                    // Write CSV row
                    writeln!(
                        file,
                        "{},{:.3},{:.1},{:.3},{},{},{},{:.8},{:.8},{:.8},{:.8},{},{:.6}",
                        k, jl_eps, cluster_radius, lambda_eps, lambda_k,
                        target_dim, aspace.n_clusters,
                        lambda_min, lambda_max, lambda_mean, lambda_spread,
                        nnz, graph_sparsity
                    )?;
                    
                    file.flush()?;
                    
                    // Early warning for degenerate cases
                    if lambda_spread < 1e-9 {
                        warn!("  ⚠ Lambda collapsed to zero (spread={:.2e})", lambda_spread);
                    }
                    if graph_sparsity > 0.98 {
                        warn!("  ⚠ Graph too sparse ({:.2}%), increase lambda_eps", graph_sparsity * 100.0);
                    }
                }
            }
        }
    }
    
    info!("Grid search complete: {}", output_path);
    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("=== Starting Sparse (Raw) Dorothea Experiment ===");

    // 1. Load Raw Sparse Data
    let data_path = PathBuf::from("./../data/DOROTHEA/dorothea_train.data");
    let n_features = 100_000; // Dorothea specific
    
    info!("Parsing sparse data from: {:?}", data_path);
    let rows = load_sparse_as_dense_normalized(&data_path, n_features)?;
    
    let n_items = rows.len();
    info!("Loaded {} items with {} features (L2 normalized)", n_items, n_features);
    
    run_grid_search(rows, "./../storage/grid_search_results.csv")?;
    
    Ok(())
}


// --- Helper: Parse Sparse Format to Dense Normalized Vec ---
fn load_sparse_as_dense_normalized(path: &PathBuf, n_features: usize) -> std::io::Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let mut row = vec![0.0; n_features];
        let mut count = 0;
        
        // Parse "12 405 99 ..." indices
        for token in line.split_whitespace() {
            if let Ok(idx) = token.parse::<usize>() {
                // Dorothea is 1-based index
                if idx > 0 && idx <= n_features {
                    row[idx - 1] = 1.0;
                    count += 1;
                }
            }
        }

        // L2 Normalization (Critical for Cosine consistency)
        if count > 0 {
            let norm = (count as f64).sqrt();
            for val in row.iter_mut() {
                *val /= norm;
            }
        } else {
            warn!("Row {} is empty (all zeros)", i);
        }
        
        dataset.push(row);
    }
    Ok(dataset)
}
