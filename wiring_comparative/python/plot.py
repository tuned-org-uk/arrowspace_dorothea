import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

# Classification results
results = pd.read_csv('dorothea_classification_results.csv')

# Spectral metrics
spectral = pd.read_csv('dorothea_spectral_metrics.csv')

# Lambda distributions
lambdas = pd.read_csv('dorothea_lambda_distributions.csv')

# UMAP embeddings (use best config)
umap_data = pd.read_csv('dorothea_umap_embeddings_gaussian_best.csv')

# ═══════════════════════════════════════════════════════════════════════════
# 2. PLOT 1: UMAP 2D EMBEDDING vs LAMBDA DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 6))

# Subplot 1: UMAP 2D embedding colored by true labels
ax1 = plt.subplot(131)
scatter = ax1.scatter(
    umap_data['umap_x'], 
    umap_data['umap_y'], 
    c=umap_data['label'], 
    cmap='RdYlBu',
    alpha=0.6,
    s=30,
    edgecolors='black',
    linewidth=0.5
)
ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
ax1.set_title('UMAP 2D Embedding\n(Colored by Drug Response)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Label (-1=Negative, +1=Positive)', fontsize=10)
ax1.grid(alpha=0.3)

# Subplot 2: Lambda distribution by class (best config)
ax2 = plt.subplot(132)
best_config = spectral.loc[spectral['cohens_d'].idxmax(), 'config']
lambda_best = lambdas[lambdas['config'] == best_config]

pos_lambdas = lambda_best[lambda_best['class'] == 'positive']['lambda']
neg_lambdas = lambda_best[lambda_best['class'] == 'negative']['lambda']

ax2.hist(pos_lambdas, bins=30, alpha=0.6, label='Positive (Drug Response)', color='#d62728', edgecolor='black')
ax2.hist(neg_lambdas, bins=30, alpha=0.6, label='Negative (No Response)', color='#1f77b4', edgecolor='black')
ax2.set_xlabel('Lambda (λ) Value', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Lambda Distribution by Class\nConfig: {best_config}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Add Cohen's d annotation
cohens_d = spectral[spectral['config'] == best_config]['cohens_d'].values[0]
ax2.text(0.05, 0.95, f"Cohen's d = {cohens_d:.3f}", 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 3: UMAP colored by train/test split
ax3 = plt.subplot(133)
for split_name in ['train', 'test']:
    split_data = umap_data[umap_data['split'] == split_name]
    ax3.scatter(
        split_data['umap_x'], 
        split_data['umap_y'], 
        label=split_name.capitalize(),
        alpha=0.5,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )
ax3.set_xlabel('UMAP Dimension 1', fontsize=12)
ax3.set_ylabel('UMAP Dimension 2', fontsize=12)
ax3.set_title('UMAP 2D Embedding\n(Train/Test Split)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('umap_vs_lambda_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: umap_vs_lambda_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PLOT 2: PERFORMANCE COMPARISON (F1 Scores)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Extract method types
results['method_type'] = results['method'].apply(lambda x: 
    'UMAP' if 'umap' in x else 
    'Lambda' if 'lambda' in x else 
    'Cosine' if 'cosine' in x else 
    'Search')

# Subplot 1: F1 Score by Method and k
ax = axes[0, 0]
for method_type in ['UMAP', 'Lambda', 'Cosine']:
    method_data = results[results['method_type'] == method_type]
    method_data['k'] = method_data['method'].str.extract(r'k(\d+)').astype(float)
    method_grouped = method_data.groupby('k')['f1'].mean()
    ax.plot(method_grouped.index, method_grouped.values, marker='o', linewidth=2, label=method_type, markersize=8)

ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('k-NN Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Subplot 2: Balanced Accuracy by Method
ax = axes[0, 1]
method_summary = results[results['method_type'].isin(['UMAP', 'Lambda', 'Cosine'])].groupby('method_type').agg({
    'balanced_accuracy': 'max',
    'f1': 'max',
    'precision': 'max',
    'recall': 'max'
})

x = np.arange(len(method_summary.index))
width = 0.2
ax.bar(x - width, method_summary['balanced_accuracy'], width, label='Balanced Acc', alpha=0.8)
ax.bar(x, method_summary['f1'], width, label='F1 Score', alpha=0.8)
ax.bar(x + width, method_summary['precision'], width, label='Precision', alpha=0.8)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Best Performance Metrics by Method', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(method_summary.index, fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0.5, 1.0])

# Subplot 3: Build Time vs F1 Score (efficiency frontier)
ax = axes[1, 0]
for method_type in ['UMAP', 'Lambda', 'Cosine']:
    method_data = results[results['method_type'] == method_type]
    best_per_config = method_data.loc[method_data.groupby('config')['f1'].idxmax()]
    ax.scatter(best_per_config['build_time_s'], best_per_config['f1'], 
               s=150, alpha=0.7, label=method_type, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Build Time (seconds)', fontsize=12)
ax.set_ylabel('Best F1 Score', fontsize=12)
ax.set_title('Efficiency Frontier: Build Time vs Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Subplot 4: Spectral Quality Metrics
ax = axes[1, 1]
spectral_metrics = ['spectral_gap', 'fiedler_value', 'effective_rank', 'cohens_d']
spectral_normalized = spectral[spectral_metrics].copy()

# Normalize to 0-1 for comparison
for col in spectral_metrics:
    max_val = spectral_normalized[col].max()
    if max_val > 0:
        spectral_normalized[col] = spectral_normalized[col] / max_val

spectral_normalized['config'] = spectral['config']
spectral_melted = spectral_normalized.melt(id_vars='config', var_name='Metric', value_name='Normalized Value')

sns.barplot(data=spectral_melted, x='Metric', y='Normalized Value', hue='config', ax=ax)
ax.set_xlabel('Spectral Metric', fontsize=12)
ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
ax.set_title('Spectral Quality Metrics by Configuration', fontsize=14, fontweight='bold')
ax.legend(title='Config', fontsize=8, title_fontsize=9, loc='upper right')
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SUMMARY: UMAP vs Lambda vs Cosine")
print("="*80)

summary = results[results['method_type'].isin(['UMAP', 'Lambda', 'Cosine'])].groupby('method_type').agg({
    'f1': ['max', 'mean'],
    'balanced_accuracy': 'max',
    'build_time_s': 'mean',
    'query_time_s': 'mean'
}).round(4)

print(summary)

# Best overall
best_method = results.loc[results['f1'].idxmax()]
print(f"\n✓ Best Overall Method: {best_method['method']} (Config: {best_method['config']})")
print(f"  F1: {best_method['f1']:.4f}, Balanced Acc: {best_method['balanced_accuracy']:.4f}")

# Lambda improvement over baselines
lambda_best_f1 = results[results['method_type'] == 'Lambda']['f1'].max()
cosine_best_f1 = results[results['method_type'] == 'Cosine']['f1'].max()
umap_best_f1 = results[results['method_type'] == 'UMAP']['f1'].max()

print(f"\nLambda vs Cosine: {((lambda_best_f1 - cosine_best_f1) / cosine_best_f1 * 100):+.1f}%")
print(f"Lambda vs UMAP:   {((lambda_best_f1 - umap_best_f1) / umap_best_f1 * 100):+.1f}%")

print("\n" + "="*80)
