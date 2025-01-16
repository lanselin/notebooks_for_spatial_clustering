"""
Utilities for spatial cluster course, winter 2025

"""

__author__ = "Luc Anselin lanselin@gmail.com,\
    Pedro Amaral pedrovma@gmail.com"


import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from libpysal.weights import KNN, w_intersection

_all_ = ["cluster_stats",
         "stress_value",
         "distcorr",
         "common_coverage"]

def cluster_stats(clustlabels):
    """
    Creates a data frame with cluster labels and cardinality

    Arguments
    ---------
    clustlabels     : cluster labels from a scikit-learn cluster class

    Returns
    -------
    clustframe      : a pandas dataframe with columns Labels and Cardinality

    """
    totclust,clustcount = np.unique(clustlabels,return_counts=True)
    cl = np.array((totclust,clustcount)).T
    clustframe = pd.DataFrame(data=cl,columns=["Labels","Cardinality"])
    return(clustframe)

def stress_value(dist,embed):
    """
    Computes the raw stress value and normalized stress value between a
    high-dimensional distance matrix and a distance matrix computed from
    embedded coordinates

    Arguments
    _________
    dist       : distance matrix in higher dimensions
    embed      : n by 2 numpy array with MDS coordinates

    Returns
    -------
    raw_stress, normalized_stress : tuple with stress values

    """
    n = dist.shape[0]
    uppind = np.triu_indices(n,k=1)
    
    reduced_distances = pairwise_distances(embed)

    distvec = dist[uppind]
    redvec = reduced_distances[uppind]

    raw_stress = np.sum((distvec - redvec) ** 2)
    denominator = np.sum(distvec ** 2)
    normalized_stress = np.sqrt(raw_stress / denominator)

    return raw_stress, normalized_stress

def distcorr(dist,embed):
    """
    Compute spearman rank correlation between upper diagonal elements
    of two distance matrices
    Uses scipy.stats.spearmanr

    Arguments
    ---------
    dist      : first distance matrix (typically higher dimension)
    embed     : n by 2 numpy array with MDS coordinates or distance
                matrix computed from coordinates

    Returns
    -------
    rho       : Spearman rank correlation

    """
    n = dist.shape[0]
    uppind = np.triu_indices(n,k=1)
    k = embed.shape[1]
    if k == 2:
        reduced_distances = pairwise_distances(embed)
    elif k == n:
        reduced_distances = embed
    else:
        raise Exception("Incompatible dimensions")

    distvec = dist[uppind]
    redvec = reduced_distances[uppind]
    rho = spearmanr(distvec,redvec)[0]
    return rho

def common_coverage(coord1,coord2,k=6):
    """
    Computes common coverage percentage between two knn weights,
    typically two MDS solutions, or geographic coordinates and MDS

    Arguments
    ---------
    coord1       : either a point geodataframe or a numpy array 
                   with coordinates
    coord2       : numpy array with coordinates (MDS)
    k            : nearest neighbor order, default = 6

    Returns
    -------
    n_int, abscov, relcov: number of non-zero overlap between two 
                           knn weights, absolute common coverage
                           percentage, relative common coverage 
                           percentage
    
    """
    # check if first argument is point layer
    if isinstance(coord1,gpd.geodataframe.GeoDataFrame):
        w1 = KNN.from_dataframe(coord1,k=k)
    elif isinstance(coord1,np.ndarray):
        w1 = KNN.from_array(coord1,k=k)
    else:
        raise Exception("Invalid input")
    w2 = KNN.from_array(coord2,k=k)
    n = coord2.shape[0]
    n_tot = n**2
    n_init = w1.nonzero
    w_int = w_intersection(w1,w2)
    n_int = w_int.nonzero
    # coverage percentages
    abscov = 100.0*n_int / n_tot
    relcov = 100.0*n_int / n_init
    return n_int, abscov, relcov

def plot_clusters(gdf,clustlabels,figsize=(5,5),title="Clusters",cmap='Set2'):
    """
    Plot clusters on a map

    Arguments
    ---------
    gdf         : geodataframe with the polygons
    clustlabels : cluster labels from a scikit-learn cluster class
    figsize     : figure size, default = (5,5)
    title       : title for the plot
    cmap        : colormap, default = 'Set2'

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt

    gdf_temp = gdf.copy()
    gdf_temp['cluster'] = clustlabels.astype(str) 

    fig, ax = plt.subplots(figsize=figsize)
    gdf_temp.plot(column='cluster', ax=ax, legend=True, cmap=cmap,
            legend_kwds={'bbox_to_anchor': (1, 0.5), 'loc': 'center left'}) 

    ax.set_title(title)
    plt.show()

def plot_dendrogram(std_data,clust_obj,labels,n_clusters,method='ward',figsize=(10,7),title="Dendrogram"):
    """
    Plot dendrogram

    Arguments
    ---------
    std_data       : standardized data
    clust_obj      : clustering object from scikit-learn
    labels         : labels for the dendrogram
    n_clusters     : number of clusters
    method         : method for linkage, default = 'ward'
    figsize        : figure size, default = (10,7)
    title          : title for the plot

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(std_data, method=method)

    # Plot the dendrogram
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, orientation='top', leaf_rotation=90, 
            leaf_font_size=7, color_threshold=clust_obj.distances_[(1-n_clusters)])
    plt.title(title)
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    plt.show()

def clusters_summary(data,clustlabels,n_clusters):
    """
    Compute the Within-cluster Sum of Squares (WSS) and Between-cluster Sum of Squares (BSS)

    Arguments
    ---------
    data         : data used for clustering
    clustlabels  : cluster labels from a scikit-learn cluster class
    n_clusters   : number of clusters

    Returns
    -------
    None

    """
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(data)

    # Compute the Total Sum of Squares (TSS) of data_cluster:
    tss = np.sum(np.square(X - X.mean(axis=0)))

    # Compute the mean of each variable by cluster
    data_tmp = data.copy().assign(cluster=clustlabels)
    cluster_means = data_tmp.groupby('cluster').mean()

    # Print the mean values
    print("Mean values by cluster:")
    print(np.round(cluster_means,2))

    # Compute the Within-cluster Sum of Squares (WSS) for each cluster
    wss_per_cluster = []
    for cluster in range(n_clusters):
        cluster_data = X[data_tmp['cluster'] == cluster]
        cluster_mean = cluster_data.mean(axis=0)
        wss = np.sum(np.square(cluster_data - cluster_mean))
        wss_per_cluster.append(wss)
    wss_per_cluster = [float(wss) for wss in wss_per_cluster]
    # Total Within-cluster Sum of Squares
    total_wss = sum(wss_per_cluster)

    # Between-cluster Sum of Squares (BSS)
    bss = tss - total_wss

    # Ratio of Between-cluster Sum of Squares to Total Sum of Squares
    ratio_bss_to_tss = bss / tss

    # Print results
    print("\nTotal Sum of Squares (TSS):", tss)
    print("Within-cluster Sum of Squares (WSS) for each cluster:", np.round(wss_per_cluster,2))
    print("Total Within-cluster Sum of Squares (WSS):", np.round(total_wss,2))
    print("Between-cluster Sum of Squares (BSS):", np.round(bss,2))
    print("Ratio of BSS to TSS:", np.round(ratio_bss_to_tss,2))

def elbow_plot(std_data, init='k-means++', max_clusters=None):
    """
    Plot the elbow plot for KMeans clustering

    Arguments
    ---------
    std_data    : standardized data
    max_clusters: maximum number of clusters to consider, default = N/5

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    if max_clusters is None:
        max_clusters = int(std_data.shape[0]/5)

    inertia = []
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, init=init, random_state=123).fit(std_data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot - Kmeans Clustering')


def plot_scatter(x, y, labels=None, title="Scatter plot", figsize=(8, 6)):
    """
    Plot a scatter plot of two variables with different colors for each cluster

    Arguments
    ---------
    x         : x-axis values
    y         : y-axis values
    labels    : cluster labels
    title     : title for the plot
    figsize   : figure size, default = (8, 6)

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    if labels is None:
        plt.scatter(x, y)
    else:
        for cluster in np.unique(labels):
            plt.scatter(
                x[labels == cluster],
                y[labels == cluster],
                label=f'Cluster {cluster}'
            )
        plt.legend(title="Clusters", fontsize=10, title_fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True)
    plt.show()

def plot_silhouette(sil_scores, obs_labels, clustlabels, title="Silhouette plot", figsize=(8, 10)):
    """
    Plot silhouette scores for each observation in each cluster

    Arguments
    ---------
    sil_scores   : silhouette scores (list)
    obs_labels   : observation labels (list)
    clustlabels  : cluster labels (list)
    title        : title for the plot
    figsize      : figure size, default = (8, 10)

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt
    import numpy as np

    silhouette_values = np.array(sil_scores) 
    observation_labels = np.array(obs_labels) 
    cluster_labels = np.array(clustlabels) 

    sorted_indices = np.lexsort((silhouette_values, cluster_labels))  
    silhouette_values_sorted = silhouette_values[sorted_indices]
    observation_labels_sorted = observation_labels[sorted_indices]
    cluster_labels_sorted = cluster_labels[sorted_indices]
    unique_clusters = np.unique(cluster_labels)
    colors = plt.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_labels_sorted == cluster
        ax.barh(
            np.arange(len(observation_labels_sorted))[cluster_mask],  
            silhouette_values_sorted[cluster_mask], 
            color=colors(i),  
            edgecolor="black",
            label=f"Cluster {cluster}"
        )

    ax.set_yticks(np.arange(len(observation_labels_sorted)))
    ax.set_yticklabels(observation_labels_sorted, fontsize=8) 
    ax.set_xlabel("Silhouette Score")
    ax.set_title(title)
    ax.axvline(x=np.mean(silhouette_values), color="red", linestyle="--", label="Mean Silhouette Score")
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()
