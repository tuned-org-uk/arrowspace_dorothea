use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::ArrowItem;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use log::{info, warn};

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

    // 2. Build ArrowSpace (No Reduction, No Noise)
    // --------------------------------------------
    // We relax 'eps' because discrete binary vectors are often orthogonal.
    // We increase 'k' to ensure graph connectivity in sparse space.
    let eps = 0.98; 
    let k = 50;     
    
    info!("Wiring Graph on RAW features (eps={}, k={})...", eps, k);
    let start = Instant::now();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(eps, k, 10, 2.0, Some(0.01)) 
        .with_inline_sampling(None)
        .with_dims_reduction(false, None) // <--- CRITICAL: No JL Projection
        .with_cluster_max_clusters(150)     // Force 150 clusters (vs auto ~11)
        .with_cluster_radius(0.85)          // Tighter clusters
        .with_persistence(
            PathBuf::from("./../storage"),
            "dorothea_raw_sparse".to_string(), // New dataset tag
        )
        .build(rows);

    info!("Build complete in {:.2?}", start.elapsed());
    
    // 3. Diagnostics
    // --------------
    info!("Graph Connected Components: (checking via simple traversal or eigen-gap hint)");
    
    // Check if we have a valid spectral spectrum
    let lambdas = aspace.lambdas();
    let min_l = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_l = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    info!("Lambda Range: {:.6} .. {:.6}", min_l, max_l);

    if min_l.abs() < 1e-9 && max_l.abs() < 1e-9 {
        warn!("WARNING: Lambdas are all zero. The graph might be fully disconnected.");
        warn!("Try increasing 'eps' (closer to 1.0) or 'k'.");
    }

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
