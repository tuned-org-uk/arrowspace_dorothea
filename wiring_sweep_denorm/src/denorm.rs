use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Strategy 1: Gaussian Noise Injection (Recommended for Dorothea)
/// Adds controlled noise to break L2=1 constraint while preserving sparsity
pub fn denormalize_gaussian_noise(
    normalized_rows: Vec<Vec<f64>>,
    noise_scale: f64,  // Recommended: 0.05-0.2
    preserve_sparsity: bool,
) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_scale).unwrap();
    
    normalized_rows.into_iter().map(|mut row| {
        let original_norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        for val in row.iter_mut() {
            if preserve_sparsity && *val == 0.0 {
                // Keep zeros sparse - only perturb non-zero values
                continue;
            }
            let noise = normal.sample(&mut rng);
            *val += noise;
            
            // Optional: ReLU to maintain non-negativity
            if *val < 0.0 {
                *val = 0.0;
            }
        }
        
        // Optional: Restore approximate original magnitude
        // This creates controlled variance while keeping scale reasonable
        let new_norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
        let scale_factor = original_norm * (1.0 + noise_scale);
        for val in row.iter_mut() {
            *val *= scale_factor / new_norm;
        }
        
        row
    }).collect()
}

/// Strategy 2: Feature-Specific Scaling (Exploit Domain Knowledge)
/// Dorothea is binary features - scale by feature frequency (inverse document frequency)
pub fn denormalize_idf_scaling(
    normalized_rows: Vec<Vec<f64>>,
    idf_weights: &[f64],  // Precomputed per-feature weights
) -> Vec<Vec<f64>> {
    normalized_rows.into_iter().map(|row| {
        row.iter()
            .zip(idf_weights.iter())
            .map(|(val, weight)| val * weight)
            .collect()
    }).collect()
}

/// Strategy 3: Power-Law Magnitude Distribution
/// Create realistic non-uniform magnitudes following Zipf's law
pub fn denormalize_powerlaw(
    normalized_rows: Vec<Vec<f64>>,
    alpha: f64,  // Power-law exponent, recommended: 1.5-2.5
) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    
    normalized_rows.into_iter().enumerate().map(|(i, row)| {
        // Assign magnitude following power law: mag_i ~ i^(-alpha)
        let rank = (i + 1) as f64;
        let magnitude = rank.powf(-alpha);
        
        row.iter().map(|val| val * magnitude).collect()
    }).collect()
}

/// Strategy 4: Additive Brownian Bridge (Smooth Perturbation)
/// Adds structured noise that preserves local smoothness
pub fn denormalize_brownian(
    normalized_rows: Vec<Vec<f64>>,
    diffusion_coeff: f64,  // Recommended: 0.01-0.1
) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, diffusion_coeff).unwrap();
    
    normalized_rows.into_iter().map(|mut row| {
        let n_features = row.len();
        let mut cumulative_noise = 0.0;
        
        for i in 0..n_features {
            if row[i] != 0.0 {  // Only perturb active features
                cumulative_noise += normal.sample(&mut rng);
                row[i] += cumulative_noise / (i + 1) as f64;  // Normalize by step
                row[i] = row[i].max(0.0);  // Keep non-negative
            }
        }
        
        row
    }).collect()
}

/// Strategy 5: Hybrid - Recommended for Dorothea
/// Combines Gaussian noise + IDF scaling for maximum spectral diversity
pub fn denormalize_hybrid(
    normalized_rows: Vec<Vec<f64>>,
    noise_scale: f64,
    feature_freq: &[usize],  // Count of non-zeros per feature
    total_docs: usize,
) -> Vec<Vec<f64>> {
    // Compute IDF weights
    let idf_weights: Vec<f64> = feature_freq
        .iter()
        .map(|&freq| {
            if freq == 0 {
                1.0
            } else {
                (total_docs as f64 / freq as f64).ln() + 1.0
            }
        })
        .collect();
    
    // First apply Gaussian noise
    let noisy = denormalize_gaussian_noise(normalized_rows, noise_scale, true);
    
    // Then apply IDF scaling
    denormalize_idf_scaling(noisy, &idf_weights)
}
