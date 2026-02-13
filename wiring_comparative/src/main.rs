use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::{ArrowItem, ArrowSpace};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;
use log::info;

// UMAP dependencies
use ndarray::{Array2, Axis};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use umap::{Umap, UmapConfig};

const K_HEAD: usize = 3;
const K_TAIL_MAX: usize = 25;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("═══════════════════════════════════════════════════════════════════════════");
    info!("  DOROTHEA RETRIEVAL EXPERIMENT v3.0");
    info!("  Tail Quality Analysis: Cosine vs Hybrid vs Taumode");
    info!("═══════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. LOAD DATA
    // ═══════════════════════════════════════════════════════════════════════════

    let data_path = PathBuf::from("./../data/DOROTHEA/dorothea_train.data");
    let label_path = PathBuf::from("./../data/DOROTHEA/dorothea_train.labels");
    let n_features = 100_000;

    info!("Loading sparse data from: {:?}", data_path);
    info!("Loading labels from: {:?}", label_path);

    let rows = load_sparse_unnormalized(&data_path, n_features)?;
    let labels = load_labels(&label_path)?;

    let n_items = rows.len();
    assert_eq!(n_items, labels.len());

    let n_positive = labels.iter().filter(|&&l| l > 0).count();
    let n_negative = labels.iter().filter(|&&l| l < 0).count();

    info!("Loaded {} items with {} features", n_items, n_features);
    info!("Labels: {} positive ({:.1}%), {} negative ({:.1}%)\n", 
          n_positive, n_positive as f64 / n_items as f64 * 100.0,
          n_negative, n_negative as f64 / n_items as f64 * 100.0);

    let split_idx = (n_items as f64 * 0.8) as usize;

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. BUILD ARROWSPACE (single best config from previous experiments)
    // ═══════════════════════════════════════════════════════════════════════════

    info!("Building ArrowSpace index...");

    let start = Instant::now();
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.8, 50, 25, 2.5, Some(0.01))
        .with_dims_reduction(true, Some(0.7))
        .with_cluster_max_clusters(50)
        .with_synthesis(arrowspace::taumode::TauMode::Median)
        .build(rows.clone());

    let build_time = start.elapsed();
    info!("✓ ArrowSpace build: {:.2?}", build_time);

    let lambdas = aspace.lambdas();
    let lambda_stats = compute_stats(lambdas);
    info!("  Lambda: μ={:.6}, σ={:.6}, range=[{:.6}, {:.6}]",
          lambda_stats.mean, lambda_stats.std, lambda_stats.min, lambda_stats.max);

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. RETRIEVAL-BASED EVALUATION (each test item = query)
    // ═══════════════════════════════════════════════════════════════════════════

    info!("\n═══════════════════════════════════════════════════════════════════════════");
    info!("  TAIL QUALITY ANALYSIS (Top-{} neighbors per query)", K_TAIL_MAX);
    info!("═══════════════════════════════════════════════════════════════════════════\n");

    let tau_configs = vec![
        ("Cosine (τ=1.0)", 1.0),
        ("Hybrid (τ=0.8)", 0.8),
        ("Taumode (τ=0.62)", 0.62),
    ];

    let train_rows = &rows[..split_idx];
    let test_rows = &rows[split_idx..];
    let train_labels = &labels[..split_idx];
    let test_labels = &labels[split_idx..];

    let mut all_search_results: Vec<QuerySearchResults> = Vec::new();
    let mut tail_metrics_records: Vec<TailMetricsRecord> = Vec::new();

    // For each test item (query)
    for (test_idx, test_row) in test_rows.iter().enumerate() {
        let query_id = test_idx + 1;
        let true_label = test_labels[test_idx];
        
        // Prepare query item with lambda
        let query = ArrowItem::new(test_row, 0.0);
        let query_lambda = aspace.prepare_query_item(&query.item, &gl);
        let query = ArrowItem::new(&query.item, query_lambda);

        let mut results_for_query: Vec<(String, Vec<(usize, f64)>)> = Vec::new();

        // Search with each tau
        for (tau_label, tau_value) in &tau_configs {
            let neighbors = aspace.search_lambda_aware(&query, K_TAIL_MAX, *tau_value);
            
            // Filter to train-only indices
            let neighbors_filtered: Vec<(usize, f64)> = neighbors
                .into_iter()
                .filter(|(idx, _)| *idx < split_idx)
                .take(K_TAIL_MAX)
                .collect();

            results_for_query.push((tau_label.to_string(), neighbors_filtered.clone()));

            // Compute tail metrics for this query+tau
            if neighbors_filtered.len() > K_HEAD {
                let tail_metrics = compute_tail_distribution(
                    &neighbors_filtered,
                    K_HEAD,
                    K_TAIL_MAX
                );

                tail_metrics_records.push(TailMetricsRecord {
                    query_id,
                    query_label: true_label,
                    tau_method: tau_label.to_string(),
                    head_mean: tail_metrics.head_mean,
                    tail_mean: tail_metrics.tail_mean,
                    tail_std: tail_metrics.tail_std,
                    tail_to_head_ratio: tail_metrics.tail_to_head_ratio,
                    tail_cv: tail_metrics.tail_cv,
                    tail_decay_rate: tail_metrics.tail_decay_rate,
                    n_tail_items: tail_metrics.n_tail_items,
                    total_items: tail_metrics.total_items,
                });
            }
        }

        all_search_results.push(QuerySearchResults {
            query_id,
            query_label: true_label,
            results_by_tau: results_for_query,
        });

        if (query_id % 20) == 0 {
            info!("  Processed {} / {} queries", query_id, test_rows.len());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. AGGREGATE TAIL METRICS
    // ═══════════════════════════════════════════════════════════════════════════

    info!("\n┌─ TAIL QUALITY SUMMARY ──────────────────────────────────────────");
    info!("│");
    info!("│ Metric            │  Cosine   │  Hybrid   │  Taumode");
    info!("│ ──────────────────────────────────────────────────────────────");

    for tau_label in tau_configs.iter().map(|(l, _)| l) {
        let tau_records: Vec<&TailMetricsRecord> = tail_metrics_records
            .iter()
            .filter(|r| &r.tau_method == tau_label)
            .collect();

        if tau_records.is_empty() {
            continue;
        }

        let avg_ratio: f64 = tau_records.iter().map(|r| r.tail_to_head_ratio).sum::<f64>()
            / tau_records.len() as f64;
        let avg_cv: f64 = tau_records.iter().map(|r| r.tail_cv).sum::<f64>()
            / tau_records.len() as f64;
        let avg_decay: f64 = tau_records.iter().map(|r| r.tail_decay_rate).sum::<f64>()
            / tau_records.len() as f64;

        info!("│ T/H Ratio         │  {:.4}  │          │         ", avg_ratio);
        info!("│ Tail CV           │  {:.4}  │          │         ", avg_cv);
        info!("│ Tail Decay        │  {:.6}│          │         ", avg_decay);
    }
    info!("│");
    info!("│ → Higher T/H ratio = better tail quality");
    info!("│ → Lower CV = more stable tail");
    info!("│ → Lower decay = flatter tail");
    info!("└─────────────────────────────────────────────────────────────────────");

    // Properly formatted aggregate table
    let mut summary_by_tau: HashMap<String, TailStats> = HashMap::new();
    
    for tau_label in tau_configs.iter().map(|(l, _)| l.to_string()) {
        let tau_records: Vec<&TailMetricsRecord> = tail_metrics_records
            .iter()
            .filter(|r| r.tau_method == tau_label)
            .collect();

        if !tau_records.is_empty() {
            summary_by_tau.insert(tau_label.clone(), TailStats {
                avg_ratio: tau_records.iter().map(|r| r.tail_to_head_ratio).sum::<f64>()
                    / tau_records.len() as f64,
                avg_cv: tau_records.iter().map(|r| r.tail_cv).sum::<f64>()
                    / tau_records.len() as f64,
                avg_decay: tau_records.iter().map(|r| r.tail_decay_rate).sum::<f64>()
                    / tau_records.len() as f64,
            });
        }
    }

    info!("\n┌─ AGGREGATE TAIL METRICS ────────────────────────────────────────");
    info!("│");
    info!("│ Method            │ T/H Ratio │ Tail CV   │ Decay Rate");
    info!("│ ──────────────────────────────────────────────────────────────");
    
    for (tau_label, _) in &tau_configs {
        if let Some(stats) = summary_by_tau.get(*tau_label) {
            info!("│ {:<17} │   {:.4}   │  {:.4}   │  {:.6}",
                  tau_label, stats.avg_ratio, stats.avg_cv, stats.avg_decay);
        }
    }
    info!("└─────────────────────────────────────────────────────────────────────");

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. SAVE CSV OUTPUTS
    // ═══════════════════════════════════════════════════════════════════════════

    info!("\n═══════════════════════════════════════════════════════════════════════════");
    info!("  SAVING RESULTS");
    info!("═══════════════════════════════════════════════════════════════════════════");

    save_tail_metrics(&tail_metrics_records, &PathBuf::from("dorothea_tail_metrics.csv"))?;
    info!("✓ dorothea_tail_metrics.csv");

    save_search_results(&all_search_results, train_labels, &PathBuf::from("dorothea_search_results.csv"))?;
    info!("✓ dorothea_search_results.csv");

    save_summary(&summary_by_tau, &PathBuf::from("dorothea_tail_summary.csv"))?;
    info!("✓ dorothea_tail_summary.csv");

    info!("\n✓ Experiment complete!");
    info!("  Compare with CVE results using the same tail metrics.");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL ANALYSIS STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct TailMetrics {
    head_mean: f64,
    tail_mean: f64,
    tail_std: f64,
    tail_to_head_ratio: f64,
    tail_cv: f64,
    tail_decay_rate: f64,
    n_tail_items: usize,
    total_items: usize,
}

#[derive(Debug, Clone)]
struct TailMetricsRecord {
    query_id: usize,
    query_label: i32,
    tau_method: String,
    head_mean: f64,
    tail_mean: f64,
    tail_std: f64,
    tail_to_head_ratio: f64,
    tail_cv: f64,
    tail_decay_rate: f64,
    n_tail_items: usize,
    total_items: usize,
}

#[derive(Debug, Clone)]
struct QuerySearchResults {
    query_id: usize,
    query_label: i32,
    results_by_tau: Vec<(String, Vec<(usize, f64)>)>,
}

#[derive(Debug, Clone)]
struct TailStats {
    avg_ratio: f64,
    avg_cv: f64,
    avg_decay: f64,
}

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL COMPUTATION (matching CVE logic)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_tail_distribution(
    results: &[(usize, f64)],
    k_head: usize,
    k_tail: usize,
) -> TailMetrics {
    let total_items = results.len().min(k_tail);
    
    if total_items <= k_head {
        return TailMetrics {
            head_mean: 0.0,
            tail_mean: 0.0,
            tail_std: 0.0,
            tail_to_head_ratio: 0.0,
            tail_cv: 0.0,
            tail_decay_rate: 0.0,
            n_tail_items: 0,
            total_items,
        };
    }

    let head_scores: Vec<f64> = results[..k_head].iter().map(|(_, s)| *s).collect();
    let tail_scores: Vec<f64> = results[k_head..total_items].iter().map(|(_, s)| *s).collect();

    let head_mean = head_scores.iter().sum::<f64>() / head_scores.len() as f64;
    let tail_mean = tail_scores.iter().sum::<f64>() / tail_scores.len() as f64;
    
    let tail_variance = tail_scores.iter()
        .map(|s| (s - tail_mean).powi(2))
        .sum::<f64>() / tail_scores.len() as f64;
    let tail_std = tail_variance.sqrt();

    let tail_to_head_ratio = if head_mean.abs() > 1e-10 {
        tail_mean / head_mean
    } else {
        0.0
    };

    let tail_cv = if tail_mean.abs() > 1e-10 {
        tail_std / tail_mean
    } else {
        0.0
    };

    let tail_decay_rate = if tail_scores.len() > 1 {
        (tail_scores[0] - tail_scores[tail_scores.len() - 1]) / tail_scores.len() as f64
    } else {
        0.0
    };

    TailMetrics {
        head_mean,
        tail_mean,
        tail_std,
        tail_to_head_ratio,
        tail_cv,
        tail_decay_rate,
        n_tail_items: tail_scores.len(),
        total_items,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSV EXPORTS
// ═══════════════════════════════════════════════════════════════════════════

fn save_tail_metrics(
    records: &[TailMetricsRecord],
    path: &PathBuf
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "query_id,query_label,tau_method,head_mean,tail_mean,tail_std,tail_to_head_ratio,tail_cv,tail_decay_rate,n_tail_items,total_items")?;
    
    for r in records {
        writeln!(file, "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                 r.query_id, r.query_label, r.tau_method, r.head_mean, r.tail_mean,
                 r.tail_std, r.tail_to_head_ratio, r.tail_cv, r.tail_decay_rate,
                 r.n_tail_items, r.total_items)?;
    }
    
    Ok(())
}

fn save_search_results(
    all_results: &[QuerySearchResults],
    train_labels: &[i32],
    path: &PathBuf
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "query_id,query_label,tau_method,rank,neighbor_idx,neighbor_label,score")?;
    
    for qr in all_results {
        for (tau_method, results) in &qr.results_by_tau {
            for (rank, (neighbor_idx, score)) in results.iter().enumerate() {
                let neighbor_label = train_labels[*neighbor_idx];
                writeln!(file, "{},{},{},{},{},{},{:.6}",
                         qr.query_id, qr.query_label, tau_method,
                         rank + 1, neighbor_idx, neighbor_label, score)?;
            }
        }
    }
    
    Ok(())
}

fn save_summary(
    summary: &HashMap<String, TailStats>,
    path: &PathBuf
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "tau_method,avg_tail_to_head_ratio,avg_tail_cv,avg_tail_decay_rate")?;
    
    let mut sorted_keys: Vec<&String> = summary.keys().collect();
    sorted_keys.sort();
    
    for key in sorted_keys {
        if let Some(stats) = summary.get(key) {
            writeln!(file, "{},{:.6},{:.6},{:.6}",
                     key, stats.avg_ratio, stats.avg_cv, stats.avg_decay)?;
        }
    }
    
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS (unchanged from original)
// ═══════════════════════════════════════════════════════════════════════════

fn load_sparse_unnormalized(path: &PathBuf, n_features: usize) -> std::io::Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut row = vec![0.0; n_features];

        for token in line.split_whitespace() {
            if let Ok(idx) = token.parse::<usize>() {
                if idx > 0 && idx <= n_features {
                    row[idx - 1] = 1.0;
                }
            }
        }
        dataset.push(row);
    }
    Ok(dataset)
}

fn load_labels(path: &PathBuf) -> std::io::Result<Vec<i32>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut labels = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let label: i32 = line.trim().parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        labels.push(label);
    }
    Ok(labels)
}

fn compute_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    Stats { mean, std, min, max }
}
