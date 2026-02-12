use arrowspace::builder::ArrowSpaceBuilder;
use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::path::PathBuf;
use std::io::BufRead;
use log::{warn, info};

use wiring_sweep_denorm::denorm::*;

pub fn run_grid_search(
    dataset: Vec<Vec<f64>>, 
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    
    // UPDATED: Reduced k sweep - focus on values that showed best performance
    let k_sweep = vec![25, 50, 100];
    
    // UPDATED: Start from 0.50 - projection is fast enough and 0.25 adds unnecessary time
    let jl_epsilons = vec![0.50, 0.60, 0.70];
    
    // UPDATED: Focus on radii that ensure good cluster assignment (>= 1.6)
    // Added intermediate values for fine-tuning
    let cluster_radii = vec![1.6, 2.0, 2.5, 3.0];
    
    // CRITICAL UPDATE: Extended lambda_eps range to combat extreme sparsity
    // Previous max 1.2 gave only 4.5% density - need much higher values
    let lambda_eps_sweep = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
    
    // UPDATED: Increase lambda_k for denser local connectivity
    let lambda_k_sweep = vec![20, 30, 50];
    
    let total = k_sweep.len() * jl_epsilons.len() * cluster_radii.len() 
                * lambda_eps_sweep.len() * lambda_k_sweep.len();
    
    // Updated CSV header to include variable lambda_k
    let mut file = File::create(output_path)?;
    writeln!(
        file, 
        "experiment,k,jl_eps,cluster_radius,lambda_eps,lambda_k,jl_dim,actual_clusters,items_assigned,lambda_min,lambda_max,lambda_mean,lambda_spread,graph_nnz,graph_sparsity,graph_density,build_time_s"
    )?;
    
    let mut count = 0;
    
    for &k in &k_sweep {
        for &jl_eps in &jl_epsilons {
            // Compute JL target dimension based on k (number of clusters)
            let jl_dim = ((8.0 * (k as f64).ln()) / (jl_eps * jl_eps)).ceil() as usize;
            let target_dim = jl_dim.min(2000).max(32); // Clamp to reasonable range
            
            for &cluster_radius in &cluster_radii {
                for &lambda_eps in &lambda_eps_sweep {
                    for &lambda_k in &lambda_k_sweep {
                        count += 1;
                        info!(
                            "Experiment {}/{}: k={}, jl_eps={:.2}, radius={:.1}, lambda_eps={:.2}, lambda_k={}", 
                            count, total, k, jl_eps, cluster_radius, lambda_eps, lambda_k
                        );
                        
                        let start_time = std::time::Instant::now();
                        
                        let (aspace, gl) = ArrowSpaceBuilder::new()
                            .with_dims_reduction(true, Some(jl_eps))
                            .with_cluster_max_clusters(k)
                            .with_cluster_radius(cluster_radius)
                            // CRITICAL FIX: Higher lambda_eps and lambda_k for denser graphs
                            .with_lambda_graph(
                                lambda_eps,      // INCREASED: now 1.5-5.0 instead of 0.5-1.2
                                lambda_k,        // INCREASED: now 20-50 instead of fixed 10
                                lambda_k/2,      // topk: results to return
                                2.0,             // p: kernel exponent
                                None             // sigma: use eps as default
                            )
                            .build(dataset.clone());
                        
                        let build_time = start_time.elapsed().as_secs_f64();
                        
                        // Extract metrics
                        let lambdas = aspace.lambdas();
                        let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let lambda_max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
                        let lambda_spread = lambda_max - lambda_min;
                        let nnz = gl.nnz();
                        let graph_shape = gl.shape();
                        let graph_sparsity = 1.0 - (nnz as f64 / (graph_shape.0 * graph_shape.1) as f64);
                        let graph_density = nnz as f64 / (graph_shape.0 * graph_shape.1) as f64;
                        
                        // Write CSV row
                        writeln!(
                            file,
                            "{},{},{:.3},{:.1},{:.2},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{},{:.6},{:.6},{:.2}",
                            count, k, jl_eps, cluster_radius, lambda_eps, lambda_k,
                            target_dim, aspace.n_clusters, 
                            dataset.len(), // items_assigned - track if all items get assigned
                            lambda_min, lambda_max, lambda_mean, lambda_spread,
                            nnz, graph_sparsity, graph_density, build_time
                        )?;
                        
                        file.flush()?;
                        
                        // Updated warnings with better thresholds
                        if lambda_spread < 1e-6 {
                            warn!("  ⚠ Lambda collapsed (spread={:.2e}), increase lambda_eps or lambda_k", lambda_spread);
                        }
                        if graph_sparsity > 0.95 {
                            warn!("  ⚠ Graph too sparse ({:.2}%), increase lambda_eps or lambda_k", graph_sparsity * 100.0);
                        } else if graph_density >= 0.10 {
                            info!("  ✓ Good density: {:.2}% ({} edges)", graph_density * 100.0, nnz);
                        }
                        
                        // Log progress for slow builds
                        if build_time > 60.0 {
                            warn!("  ⚠ Slow build: {:.1}s (consider reducing jl_eps)", build_time);
                        }
                    }
                }
            }
        }
    }
    
    info!("Grid search complete: {}", output_path);
    info!("Total experiments: {} (estimated time: {:.1} min)", total, total as f64 * 0.5);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("=== Denormalized Dorothea Experiment ===");
    info!("Testing ArrowSpace on non-normalized sparse data");

    let data_path = PathBuf::from("./../data/DOROTHEA/dorothea_train.data");
    let n_features = 100_000;
    
    // Load normalized baseline
    info!("Loading and normalizing baseline data...");
    let normalized_rows = load_sparse_as_dense_normalized(&data_path, n_features)?;
    
    // Compute feature frequencies for IDF
    info!("Computing feature statistics for denormalization...");
    let feature_freq = compute_feature_frequencies(&normalized_rows);
    let total_docs = normalized_rows.len();
    
    // Test multiple denormalization strategies
    let strategies = vec![
        ("normalized_baseline", normalized_rows.clone()),
        ("gaussian_005", denormalize_gaussian_noise(normalized_rows.clone(), 0.05, true)),
        ("gaussian_010", denormalize_gaussian_noise(normalized_rows.clone(), 0.10, true)),
        ("gaussian_020", denormalize_gaussian_noise(normalized_rows.clone(), 0.20, true)),
        ("idf_only", denormalize_idf_scaling(normalized_rows.clone(), &compute_idf(&feature_freq, total_docs))),
        ("hybrid_005", denormalize_hybrid(normalized_rows.clone(), 0.05, &feature_freq, total_docs)),
        ("hybrid_010", denormalize_hybrid(normalized_rows.clone(), 0.10, &feature_freq, total_docs)),
        ("powerlaw_15", denormalize_powerlaw(normalized_rows.clone(), 1.5)),
        ("powerlaw_20", denormalize_powerlaw(normalized_rows.clone(), 2.0)),
    ];
    
    for (name, data) in strategies {
        info!("Running experiment: {}", name);
        
        // Log magnitude statistics
        let norms: Vec<f64> = data.iter()
            .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        let mean_norm = norms.iter().sum::<f64>() / norms.len() as f64;
        let std_norm = (norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f64>() 
                        / norms.len() as f64).sqrt();
        
        info!("  Magnitude stats: mean={:.4}, std={:.4}, cv={:.4}", 
              mean_norm, std_norm, std_norm / mean_norm);
        
        let output_path = format!("./../storage/denorm_sweep_results_{}.csv", name);
        run_grid_search(data, &output_path)?;
    }
    
    Ok(())
}

fn compute_feature_frequencies(rows: &[Vec<f64>]) -> Vec<usize> {
    let n_features = rows[0].len();
    let mut freq = vec![0; n_features];
    
    for row in rows {
        for (i, &val) in row.iter().enumerate() {
            if val.abs() > 1e-9 {
                freq[i] += 1;
            }
        }
    }
    
    freq
}

fn compute_idf(feature_freq: &[usize], total_docs: usize) -> Vec<f64> {
    feature_freq
        .iter()
        .map(|&freq| {
            if freq == 0 {
                1.0
            } else {
                (total_docs as f64 / freq as f64).ln() + 1.0
            }
        })
        .collect()
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
    
    info!("Loaded {} rows from sparse format", dataset.len());
    Ok(dataset)
}
