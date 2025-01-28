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
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

_all_ = ["cluster_stats",
         "stress_value",
         "distcorr",
         "common_coverage",
         "plot_dendrogram",
         "cluster_center",
         "cluster_fit",
         "cluster_map",
         "elbow_plot",
         "plot_silhouette"]


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


def cluster_map(gdf, clustlabels, title='Clusters', grid_shape=(1, 1), figsize=(5, 5), cmap='Set2', show_axis=False):
    """
    Plot multiple cluster maps in a grid. Can handle both single and multiple maps.

    Arguments
    ---------
    gdf          : geodataframe with the polygons
    clustlabels  : list or single array of cluster labels
    title        : list or single string of titles for each subplot
    grid_shape   : tuple defining the grid layout (default = (1,1))
    figsize      : figure size, default = (5,5)
    cmap         : colormap, default = 'Set2'
    show_axis    : flag to show axis, default = False

    Returns
    -------
    None
    """
    if not isinstance(clustlabels, (list, tuple)):
        clustlabels = [clustlabels]
    if not isinstance(title, (list, tuple)):
        title = [title]
    
    num_maps = len(clustlabels) 
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = np.array(axes).flatten()
    
    for i in range(num_maps):
        gdf_temp = gdf.copy()  
        gdf_temp['cluster'] = np.array(clustlabels[i]).astype(str) 
        
        gdf_temp.plot(column='cluster', ax=axes[i], legend=True, cmap=cmap,
                      legend_kwds={'bbox_to_anchor': (1, 0.5), 'loc': 'center left'})
        
        if not show_axis:
            axes[i].axis('off') 
        
        axes[i].set_title(title[i]) 
    
    plt.tight_layout()  
    plt.show()  



def plot_dendrogram(std_data,n_clusters,
                    package='scipy',method='ward',
                    labels=None,figsize=(12,7),title="Dendrogram"):
    """
    Plot dendrogram

    Arguments
    ---------
    std_data       : standardized data or linkage result from scipy.cluster
    n_clusters     : number of clusters
    package        : module used for cluster calculation, default `scipy`, linkage
                     structure is passed as std_data, option `scikit` computes
                     linkage from standardize input array
    labels         : labels for the dendrogram, default None, uses sequence numbers,
                     otherwise numpy array (typically taken from original data frame)
    method         : method for linkage, default = 'ward', ignored when linkage is passed
    figsize        : figure size, default = (12,7)
    title          : title for the plot, default "Dendrogram"

    Returns
    -------
    R              : dictionary produced by dendrogram
    """
    nclusters = n_clusters
    if package == 'scikit':
        Z = linkage(std_data, method=method)
    elif package == 'scipy':
        Z = std_data
    else:
        raise Exception("Invalid input")

    # Plot the dendrogram
    plt.figure(figsize=figsize)
    R = dendrogram(Z, labels=labels, orientation='top', leaf_rotation=90, 
            leaf_font_size=7, color_threshold=Z[1-nclusters,2])
    plt.title(title)
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    plt.show()
    return R

def cluster_center(data,clustlabels):
    """
    Compute cluster centers for original variables

    Arguments
    ---------
    data         : data frame with cluster variable observations
    clustlabels  : cluster labels (integer or string)

    Returns
    -------
    clust_means,clust_medians : tuple with data frames of cluster means
                                and cluster medians for each variable
    """

    dt_clust = data.copy().assign(cluster=clustlabels)
    clust_means = dt_clust.groupby('cluster').mean()
    clust_medians = dt_clust.groupby('cluster').median()
    return clust_means,clust_medians

def cluster_fit(data,clustlabels,n_clusters,correct=False,printopt=True):
    """
    Compute the sum of squared deviations from the mean measures of fit.

    Arguments
    ---------
    data         : data used for clustering
    clustlabels  : cluster labels
    n_clusters   : number of clusters
    correct      : correction for degrees of freedom, default = False for
                   no correction (division by n), other option is True, 
                   which gives division by n-1
    printopt     : flag to provide listing of results, default = True

    Returns
    -------
    clustfit     : dictionary with fit results
                   TSS = total sum of squares
                   Cluster_WSS = WSS per cluster
                   WSS = total WSS
                   BSS = total BSS
                   Ratio = BSS/WSS
    """

    clustfit = {}

    X = StandardScaler().fit_transform(data)
    if correct:
        n = X.shape[0]
        nn = np.sqrt((n - 1.0)/n)
        X = X * nn
    # Compute the Total Sum of Squares (TSS) of data_cluster:
    #tss = np.sum(np.square(X - X.mean(axis=0)))
    tss = np.sum(np.square(X))  # X is standardized, mean = 0
    clustfit["TSS"] = tss
    # Compute the mean of each variable by cluster
    data_tmp = data.copy().assign(cluster=clustlabels)
    #cluster_means = data_tmp.groupby('cluster').mean()

    # Compute the Within-cluster Sum of Squares (WSS) for each cluster
    wss_per_cluster = []
    for cluster in range(n_clusters):
        cluster_data = X[data_tmp['cluster'] == cluster]
        if cluster_data.shape[0] > 1: # avoid issues with singletons
            cluster_mean = cluster_data.mean(axis=0)
            wss = np.sum(np.square(cluster_data - cluster_mean))
        else:
            wss = 0.0
        wss_per_cluster.append(wss)
    wss_per_cluster = [float(wss) for wss in wss_per_cluster]
    clustfit["Cluster_WSS"] = wss_per_cluster
    # Total Within-cluster Sum of Squares
    total_wss = sum(wss_per_cluster)
    clustfit["WSS"] = total_wss
    # Between-cluster Sum of Squares (BSS)
    bss = tss - total_wss
    clustfit["BSS"] = bss
    # Ratio of Between-cluster Sum of Squares to Total Sum of Squares
    ratio_bss_to_tss = bss / tss
    clustfit["Ratio"] = ratio_bss_to_tss
    if printopt:
        # Print results
        print("\nTotal Sum of Squares (TSS):", tss)
        print("Within-cluster Sum of Squares (WSS) for each cluster:", np.round(wss_per_cluster,3))
        print("Total Within-cluster Sum of Squares (WSS):", np.round(total_wss,3))
        print("Between-cluster Sum of Squares (BSS):", np.round(bss,3))
        print("Ratio of BSS to TSS:", np.round(ratio_bss_to_tss,3))
    return clustfit


def elbow_plot(std_data, n_init = 150, init='k-means++', max_clusters=20,
               random_state= 1234567):
    """
    Plot the elbow plot for partitioning clustering methods

    Arguments
    ---------
    std_data    : standardized data
    n_init      : number of inital runs, default 150
    init        : K-means initialization, default = 'k-means++'
    max_clusters: maximum number of clusters to consider, default = 20

    Returns
    -------
    None
    """

    inertia = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init=n_init, init=init, random_state=random_state).fit(std_data)
        inertia.append(kmeans.inertia_)
    plt.plot(range(2, max_clusters+1), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.xticks(range(2, max_clusters+1, 2))
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')


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


def plot_silhouette(sil_scores, obs_labels, clustlabels, 
                    title="Silhouette plot", figsize=(8, 10), font_size = 8):
    """
    Plot silhouette scores for each observation in each cluster

    Arguments
    ---------
    sil_scores   : silhouette scores (list)
    obs_labels   : observation labels (list)
    clustlabels  : cluster labels (list)
    title        : title for the plot
    figsize      : figure size, default = (8, 10)
    fontsize     : size for label, default = 8

    Returns
    -------
    None
    """

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
    ax.set_yticklabels(observation_labels_sorted, fontsize=font_size) 
    ax.set_xlabel("Silhouette Score")
    ax.set_title(title)
    ax.axvline(x=np.mean(silhouette_values), color="red", linestyle="--", label="Mean Silhouette Score")
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

