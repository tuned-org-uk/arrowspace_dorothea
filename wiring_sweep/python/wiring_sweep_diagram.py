import pandas as pd
import plotly.express as px
from pathlib import Path

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / "storage/004_grid_search_results_v2.csv")

# Derive density
df["graph_density"] = 1.0 - df["graph_sparsity"]

# 1. Heatmap: density vs lambda_eps & jl_eps
heat1 = df.groupby(["jl_eps", "lambda_eps"])["graph_density"].mean().reset_index()
fig1 = px.density_heatmap(
    heat1,
    x="lambda_eps",
    y="jl_eps",
    z="graph_density",
    color_continuous_scale="Viridis",
    title="Graph density vs lambda_eps and jl_eps"
)
fig1.update_xaxes(title="lambda_eps")
fig1.update_yaxes(title="jl_eps")
fig1.show()

# 2. Line: density vs lambda_eps by k
line1 = df.groupby(["k", "lambda_eps"])["graph_density"].mean().reset_index()
fig2 = px.line(
    line1,
    x="lambda_eps",
    y="graph_density",
    color="k",
    markers=True,
    title="Graph density vs lambda_eps by k"
)
fig2.update_xaxes(title="lambda_eps")
fig2.update_yaxes(title="graph_density")
fig2.show()

# 3. Heatmap: density vs lambda_eps & cluster_radius
heat2 = df.groupby(["cluster_radius", "lambda_eps"])["graph_density"].mean().reset_index()
fig3 = px.density_heatmap(
    heat2,
    x="lambda_eps",
    y="cluster_radius",
    z="graph_density",
    color_continuous_scale="Viridis",
    title="Graph density vs lambda_eps and cluster_radius"
)
fig3.update_xaxes(title="lambda_eps")
fig3.update_yaxes(title="cluster_radius")
fig3.show()

# 4. Scatter: lambda_mean vs graph_density colored by k
fig4 = px.scatter(
    df,
    x="graph_density",
    y="lambda_mean",
    color="k",
    facet_col="jl_eps",
    title="lambda_mean vs graph density by k and jl_eps"
)
fig4.update_xaxes(title="graph_density")
fig4.update_yaxes(title="lambda_mean")
fig4.show()
