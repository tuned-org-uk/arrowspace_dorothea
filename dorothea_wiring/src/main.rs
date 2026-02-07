use arrowspace::builder::ArrowSpaceBuilder;
use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;
use serde::Deserialize; // FIX: Import for local struct
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

// Create a local struct to handle JSON loading since GraphParams
// in 0.24.6 does not implement DeserializeOwned [file:3]
#[derive(Deserialize)]
struct EstimatedParams {
    eps: f64,
    k: usize,
    topk: usize,
    p: f64,
    sigma: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the dense high-dim matrix
    let x: Array2<f64> = read_npy("./../storage/dorothea_highdim_full100k.npy")?;

    // 2. Load the JSON exported from Python
    let file = File::open("./../storage/estimated_graph_params.json")?;
    let reader = BufReader::new(file);
    let params: EstimatedParams = serde_json::from_reader(reader)?;

    // 3. Convert Array2 to Vec<Vec<f64>> efficiently
    // To minimize memory spikes, we consume the rows into the nested vector format
    println!("Preparing rows for ArrowSpace core...");
    let rows: Vec<Vec<f64>> = x.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();

    // Explicitly drop the original ndarray to free memory before build starts
    drop(x);

    println!(
        "Loaded params: eps={}, k={}, sigma={:?}",
        params.eps, params.k, params.sigma
    );
    println!("Wiring ArrowSpace graph (Max-Stress Build)...");

    let start = Instant::now();

    // 3. Build ArrowSpace
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(params.eps, params.k, params.topk, params.p, params.sigma)
        .with_inline_sampling(None)
        .with_dims_reduction(true, Some(params.eps))
        .with_persistence(
            PathBuf::from("./../storage"),
            "dorothea_highdim".to_string(),
        )
        .build(rows);

    println!("Build complete in {:?}", start.elapsed());
    println!("Graph Nodes: {}, Shape: {:?}", gl.nnodes, gl.shape());

    Ok(())
}
