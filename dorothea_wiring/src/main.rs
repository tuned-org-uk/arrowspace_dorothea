use arrowspace::builder::ArrowSpaceBuilder;
use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

// Logging crates
use log::{info, warn, debug, trace};

#[derive(Deserialize, Debug)]
struct EstimatedParams {
    eps: f64,
    k: usize,
    topk: usize,
    p: f64,
    sigma: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Logging Harness
    // Set RUST_LOG=info or RUST_LOG=debug to see ArrowSpace's internal traces [file:3]
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting Dorothea High-Dimensional Build Pipeline");

    // 2. Load the dense high-dim matrix
    let npy_path = "./../storage/dorothea_highdim_full100k.npy";
    debug!("Reading NPY from: {}", npy_path);
    let x: Array2<f64> = read_npy(npy_path)?;
    info!("Matrix loaded: {} items x {} features", x.nrows(), x.ncols());

    // 3. Load the JSON exported from Python
    let json_path = "./../storage/estimated_graph_params.json";
    let file = File::open(json_path)?;
    let reader = BufReader::new(file);
    let params: EstimatedParams = serde_json::from_reader(reader)?;
    info!("Loaded spectral parameters: {:?}", params);

    // 4. Convert Array2 to Vec<Vec<f64>>
    trace!("Commencing row conversion Axis(0) -> Vec<Vec<f64>>");
    let rows: Vec<Vec<f64>> = x.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();

    // Freeing large memory block before building the graph
    let memory_freed = (x.len() * 8) / (1024 * 1024);
    drop(x);
    debug!("Original ndarray dropped (approx {} MB freed)", memory_freed);

    // 5. Build ArrowSpace
    info!("Wiring Laplacian Graph (Max-Stress Mode)...");
    let start = Instant::now();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(params.eps, params.k, params.topk, params.p, params.sigma)
        .with_inline_sampling(None)
        .with_dims_reduction(true, Some(params.eps)) // Internal JL harness [file:3]
        .with_persistence(
            PathBuf::from("./../storage"),
            "dorothea_highdim".to_string(),
        )
        .build(rows);

    let duration = start.elapsed();
    info!("ArrowSpace Build Success in {:.2?}", duration);

    // 6. Validation Output
    info!("Final Graph Stats:");
    info!("  - Nodes:     {}", gl.nnodes);
    info!("  - Laplacian: {:?}", gl.shape());
    info!("  - Items:     {}", aspace.nitems);

    if aspace.nitems == 0 {
        warn!("Build completed but ArrowSpace contains 0 items. Check row conversion.");
    }

    Ok(())
}
