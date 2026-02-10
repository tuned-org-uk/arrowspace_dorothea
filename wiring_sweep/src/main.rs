use arrowspace::builder::ArrowSpaceBuilder;
use std::fs::File;
use std::io::Write;
use log::info;

pub fn run_grid_search(
    dataset: Vec<Vec<f64>>, 
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    
    let k_sweep = vec![5, 10, 20, 50, 100, 150, 200, 300];
    let epsilons = vec![0.1, 0.2, 0.3, 0.5];
    let radii = vec![0.5, 0.7, 1.0, 1.5, 2.0];
    
    let total = k_sweep.len() * epsilons.len() * radii.len();
    
    // Create CSV file
    let mut file = File::create(output_path)?;
    writeln!(file, "k,epsilon,radius_mult,jl_dim,actual_clusters,lambda_min,lambda_max,lambda_mean,graph_sparsity,graph_nnz")?;
    
    let mut count = 0;
    
    for &k in &k_sweep {
        for &eps in &epsilons {
            let jl_dim = ((8.0 * (k as f64).ln()) / (eps * eps)).ceil() as usize;
            let target_dim = jl_dim.min(2000);
            
            for &radius_mult in &radii {
                count += 1;
                info!("Experiment {}/{}: k={}, eps={:.2}, radius={:.1}", count, total, k, eps, radius_mult);
                
                let (aspace, gl) = ArrowSpaceBuilder::new()
                    .with_dims_reduction(true, Some(eps))
                    .with_cluster_max_clusters(Some(k))
                    .with_cluster_radius(1.0 * radius_mult)
                    .with_lambda_graph(1e-3, 6, 3, 2.0, None)
                    .build(dataset.clone());
                
                // Extract metrics
                let lambdas = aspace.lambdas();
                let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let lambda_max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
                let sparsity = gl.sparsity();
                let nnz = gl.nnz();
                
                // Write CSV row
                writeln!(
                    file,
                    "{},{:.3},{:.1},{},{},{:.8},{:.8},{:.8},{:.6},{}",
                    k, eps, radius_mult, target_dim, aspace.nclusters,
                    lambda_min, lambda_max, lambda_mean, sparsity, nnz
                )?;
                
                file.flush()?;
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
    // -----------------------
    let data_path = PathBuf::from("./../data/DOROTHEA/dorothea_train.data");
    let n_features = 100_000; // Dorothea specific
    
    info!("Parsing sparse data from: {:?}", data_path);
    let rows = load_sparse_as_dense_normalized(&data_path, n_features)?;
    
    let n_items = rows.len();
    info!("Loaded {} items with {} features (L2 normalized)", n_items, n_features);
    
    run_grid_search(dataset, "./../storage/grid_search_results.csv")?;
    
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
        // For binary, L2 norm is sqrt(count)
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

