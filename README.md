# spatial-cluster-helper
A helper package for analyzing and visualizing spatial clusters, designed to facilitate reproducibility and ease of use in academic and applied research contexts. Developed in support of Anselin (2024).

## 📦 Features

### Cluster Analysis & Validation
- **Cluster Statistics**: Generate labels/cardinality summaries (`cluster_stats`)
- **Fit Metrics**: TSS, WSS, BSS, and cluster quality ratios (`cluster_fit`)
- **Stress Evaluation**: Raw/normalized stress values (`stress_value`)
- **Optimal Clusters**: Elbow plot visualization for K-means (`elbow_plot`)
- **Silhouette Analysis**: Observation-level silhouette scores (`plot_silhouette`)

### Spatial Analysis Tools
- **Validation Indices**:
  - Fragmentation (entropy/Simpson) (`cluster_fragmentation`)
  - Spatial autocorrelation (Join Count Ratios) (`cluster_joincount`)
  - Compactness & diameter metrics (`cluster_compactness`, `cluster_diameter`)
- **Neighborhood Overlap**: KNN coverage comparison (`common_coverage`)

### Visualization
- **Cluster Maps**: Geographic cluster visualization (`cluster_map`)
- **Dendrograms**: Hierarchical clustering trees (`plot_dendrogram`)
- **Scatter Plots**: 2D cluster visualization (`plot_scatter`)

### Utilities
- **Cluster Centers**: Mean/median descriptors (`cluster_center`)
- **Data Management**: Automated example datasets handling (`ensure_datasets`)

## 🚀 Installation

You can install the package from pypi:

```bash
pip install spatial-cluster-helper
```

## 🗂️ Usage
You can check several usage examples in the lab materials developed for the Spatial Cluster Analysis course taught at the University of Chicago in the Winter of 2025 [here](https://github.com/lanselin/notebooks_for_spatial_clustering).

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

**Developed at The Center for Spatial Data Science at the University of Chicago by Luc Anselin and Pedro Amaral**  


