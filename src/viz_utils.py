import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

# ---------------------------------------------------------------------------
# Clustering & heat‑map visualisation
# ---------------------------------------------------------------------------

def plot_clusters(df: pd.DataFrame, n_clusters: int = 9, feature_cols: list | None = None):
    """Run K‑means clustering on *df* and display a heat‑map of cluster means.

    Parameters
    ----------
    df: DataFrame
        Data containing the numerical features defined in *feature_cols*.
    n_clusters: int, default 9
        Number of clusters for K‑means.
    feature_cols: list of str, optional
        Columns to use for clustering. If omitted a sensible default set is used.
    """
    if feature_cols is None:
        feature_cols = [
            'Intelligence Index', 'Coding Index', 'Math Index (Reasoning)', 'MMLU Pro (Knowledge)',
            'Norm_Speed', 'Norm_Context', 'Norm_Cost_In', 'Norm_Cost_Out', 'Norm_Latency'
        ]
    # Drop rows with missing values for the clustering features
    df_clean = df.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_clean['Cluster_ID'] = kmeans.fit_predict(X)
    # Merge cluster IDs back into the original frame (preserve NaNs for non‑clusterable rows)
    df['Cluster_ID'] = np.nan
    df.update(df_clean[['Cluster_ID']])

    # Compute mean profile per cluster
    cluster_profiles = df_clean.groupby('Cluster_ID')[feature_cols].mean()
    # Friendly column names for the heat‑map
    pretty_names = [
        'Intelligence Index', 'Coding Index', 'Math Index (Reasoning)', 'MMLU Pro (Knowledge)',
        'Speed (Tokens/s)', 'Context', 'Cost In (Token)', 'Cost Out (Token)', 'Latency'
    ]
    cluster_profiles.columns = pretty_names

    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Cluster (K={n_clusters})")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.show()

    # Print example models per cluster
    for i in range(n_clusters):
        print(f"\n🏷️ Examples in Cluster {i}:")
        cluster_mods = df_clean[df_clean['Cluster_ID'] == i]
        name_col = 'Link' if 'Link' in cluster_mods.columns else cluster_mods.columns[0]
        display(cluster_mods[[name_col, 'Costo Input (1M Token) $', 'Intelligence Index']].head(10))

# ---------------------------------------------------------------------------
# Interactive Pareto / scatter visualisation
# ---------------------------------------------------------------------------

def interactive_scatter(df: pd.DataFrame):
    """Create an interactive Plotly scatter with Pareto front selection.

    The widget lets the user pick two metrics (X and Y). The function draws the
    full scatter, highlights the Pareto‑optimal points, and prints the count of
    models that constitute the best trade‑off.
    """
    metriche = [
        'Costo Input (1M Token) $', 'Costo Output (1M Token) $', 'Latency (s)',
        'Speed (Tokens/s)', 'Intelligence Index', 'Coding Index',
        'Math Index (Reasoning)', 'MMLU Pro (Knowledge)', 'Context Window (Token)'
    ]
    da_minimizzare = ['Costo Input (1M Token) $', 'Costo Output (1M Token) $', 'Latency (s)']
    metriche_log = ['Costo Input (1M Token) $', 'Costo Output (1M Token) $',
                    'Latency (s)', 'Speed (Tokens/s)', 'Context Window (Token)']

    out = widgets.Output()

    def _update(x_metric: str, y_metric: str):
        with out:
            clear_output(wait=True)
            if x_metric == y_metric:
                print("Select two different metrics for the axes.")
                return

            df_plot = df.copy()
            df_plot[x_metric] = pd.to_numeric(df_plot[x_metric], errors='coerce')
            df_plot[y_metric] = pd.to_numeric(df_plot[y_metric], errors='coerce')
            df_plot = df_plot.dropna(subset=[x_metric, y_metric])
            if df_plot.empty:
                print("No data for the selected metric combination.")
                return

            # Sort to make Pareto sweep deterministic
            sort_asc = x_metric in da_minimizzare
            df_plot = df_plot.sort_values(by=[x_metric, y_metric], ascending=[sort_asc, not sort_asc])

            # Pareto front extraction
            pareto = []
            best_y = float('inf') if y_metric in da_minimizzare else float('-inf')
            for _, row in df_plot.iterrows():
                y_val = row[y_metric]
                better = (y_val < best_y) if y_metric in da_minimizzare else (y_val > best_y)
                if better:
                    pareto.append(row)
                    best_y = y_val
            df_front = pd.DataFrame(pareto)

            name_col = 'Model Name (AA)' if 'Model Name (AA)' in df_plot.columns else 'Link'
            fig = px.scatter(
                df_plot,
                x=x_metric,
                y=y_metric,
                hover_name=name_col,
                title=f"{x_metric} vs {y_metric}",
                template='plotly_white',
                height=600,
                opacity=0.6
            )
            if x_metric in metriche_log:
                fig.update_layout(xaxis_type='log')
            if y_metric in metriche_log:
                fig.update_layout(yaxis_type='log')
            if not df_front.empty:
                df_front = df_front.sort_values(by=x_metric)
                fig.add_trace(go.Scatter(
                    x=df_front[x_metric],
                    y=df_front[y_metric],
                    mode='lines+markers',
                    line=dict(color='red', width=2, shape='hv'),
                    name='Optimal models',
                    marker=dict(size=8, color='red')
                ))
            fig_widget = go.FigureWidget(fig)
            display(fig_widget)
            print(f"{len(df_front)} models with best trade‑off:")
            display(df_front[[name_col, x_metric, y_metric]].reset_index(drop=True))

    dropdown_x = widgets.Dropdown(options=metriche, value=metriche[0], description='X axis:')
    dropdown_y = widgets.Dropdown(options=metriche, value=metriche[1], description='Y axis:')
    dropdown_x.observe(lambda ch: _update(dropdown_x.value, dropdown_y.value), names='value')
    dropdown_y.observe(lambda ch: _update(dropdown_x.value, dropdown_y.value), names='value')

    display(widgets.HBox([dropdown_x, dropdown_y]))
    display(out)
    # Initialise with default selection
    _update(dropdown_x.value, dropdown_y.value)
