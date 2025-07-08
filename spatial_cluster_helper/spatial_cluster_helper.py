"""
Utilities for spatial cluster analysis
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

# Specifically for the ensure_datasets function
import os
import zipfile
import requests
from io import BytesIO

_all_ = ["cluster_stats",
         "stress_value",
         "distcorr",
         "common_coverage",
         "plot_dendrogram",
         "cluster_center",
         "cluster_fit",
         "cluster_map",
         "elbow_plot",
         "plot_silhouette",
         "cluster_fragmentation",
         "cluster_joincount",
         "cluster_compactness",
         "cluster_diameter"]


def cluster_stats(clustlabels, printopt=True):
    """
    Creates a data frame with cluster labels and cardinality

    Arguments
    ---------
    clustlabels     : cluster labels from a scikit-learn cluster class
    printopt        : flag to print the results, default = True

    Returns
    -------
    clustframe      : a pandas dataframe with columns Labels and Cardinality
    """

    totclust,clustcount = np.unique(clustlabels,return_counts=True)
    cl = np.array((totclust,clustcount)).T
    clustframe = pd.DataFrame(data=cl,columns=["Labels","Cardinality"])
    if printopt:
        print(clustframe.to_string(index=False))
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


def common_coverage(coord1,coord2,k=6,silence_warnings=False):
    """
    Computes common coverage percentage between two knn weights,
    typically two MDS solutions, or geographic coordinates and MDS

    Arguments
    ---------
    coord1       : either a point geodataframe or a numpy array 
                   with coordinates
    coord2       : numpy array with coordinates (MDS)
    k            : nearest neighbor order, default = 6
    silence_warnings : flag to silence warnings in libpysal weights, default = False

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
    w_int = w_intersection(w1,w2,silence_warnings=silence_warnings)
    n_int = w_int.nonzero
    # coverage percentages
    abscov = 100.0*n_int / n_tot
    relcov = 100.0*n_int / n_init
    return n_int, abscov, relcov


def cluster_map(gdf, clustlabels, title='Clusters', grid_shape=(1, 1), figsize=(5, 5), cmap='Set2', show_axis=False, baselayer=None, markersize=10, legend_fontsize=None):
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
    baselayer    : geodataframe with the baselayer, default = None
    markersize   : size of the markers, if any, default = 10
    legend_fontsize: size of the legend font, default = None

    Returns
    -------
    None
    """
    if not isinstance(clustlabels, (list, tuple)):
        clustlabels = [clustlabels]
    if not isinstance(title, (list, tuple)):
        title = [title]
    
    legend_kwds = {'bbox_to_anchor': (1, 0.5), 'loc': 'center left'}
    if legend_fontsize is not None:
        legend_kwds['fontsize'] = legend_fontsize

    num_maps = len(clustlabels) 
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = np.array(axes).flatten()
    
    for i in range(num_maps):
        gdf_temp = gdf.copy()  
        gdf_temp['cluster'] = np.array(clustlabels[i]).astype(str) 
        if baselayer is not None:
            baselayer.plot(ax=axes[i], color=(1, 1, 1, 0), edgecolor='black')
        gdf_temp.plot(column='cluster', ax=axes[i], legend=True, cmap=cmap,
                      legend_kwds=legend_kwds, markersize=markersize)
        
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
    for cluster in set(clustlabels):
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

def cluster_fragmentation(clust_stats,cluster_fragmentation,fragmentation,spatially_constrained):
    """
    Fragmentation indices from pygeoda spatial validation

    Arguments:
    ----------
    clust_stats           : data frame with cluster cardinalities
    cluster_fragmentation : cluster_fragementation attribute from pygeoda.spatial_validation
                            entropy, standardized entropy, Simpson and standardized Simpson by cluster
    fragmentation         : fragmentation attribute from pygeoda.spatial_validation
                            same items as cluster_fragmentation but for all clusters together
    spatially_contrained  : flag for spatially_constrained from pygeoda.spatial_validation

    Returns:
    --------
    valfrag               : data frame with fragmentation statistics
    
    """
    
    frag = cluster_fragmentation
    fragall = fragmentation
    fragt = []
    ntot = clust_stats['Cardinality'].sum()
    if not(spatially_constrained):
        jj = 0
        for i in frag:
            nn = clust_stats['Cardinality'].iloc[jj]
            fragt.append({
                'Label' : jj,
                'N' : nn,
                "Sub" : i.n,
                "Entropy" : i.entropy,
                "Entropy*" : i.std_entropy,
                "Simpson" : i.simpson,
                "Simpson*" : i.std_simpson
            })
            jj = jj + 1

    fragt.append({
        'Label' : "All",
        'N' : ntot,
        'Sub' : "",
        "Entropy" : fragall.entropy,
        "Entropy*" : fragall.std_entropy,
        "Simpson" : fragall.simpson,
        "Simpson*" : fragall.std_simpson
    })

    valfrag = pd.DataFrame(fragt)

    print("Fragmentation")
    print(valfrag.to_string(index=False))

    return(valfrag)

def cluster_joincount(clust_stats,joincount_ratio,all_joincount_ratio):
    """
    Join count cluster statistics from pygeoda.spatial_validation

    Arguments:
    ----------
    clust_stats         : cluster cardinalities
    joincount_ratio     : joincount_ratio attribute from pygeoda.spatial_validation
                          join count statistics by cluster
    all_joincount_ratio : all_joincount_ration attribute from pygeoda.spatial_validation
                          join count statistics for all clusters in aggregate

    Returns:
    --------
    valjc               : data frame with join count statistics
    
    """

    jc = joincount_ratio
    ntot = clust_stats['Cardinality'].sum()
    joinct = []
    jj = 0
    nbrstot = 0
    jctot = 0
    for i in jc:
        nbrstot = nbrstot + i.neighbors
        jctot = jctot + i.join_count
        joinct.append({
            'Label' : jj,
            "N" : i.n,
            "Neighbors" : i.neighbors,
            "Join Count" : i.join_count,
            "Ratio" : np.round(i.ratio,3)
        })
        jj = jj + 1
    joinct.append({
        'Label' : "All",
        'N' : ntot,
        "Neighbors" : nbrstot,
        "Join Count" : jctot,
        "Ratio" : np.round(all_joincount_ratio.ratio,4)
    })

    valjc = pd.DataFrame(joinct)

    print("Join Count Ratio")
    print(valjc.to_string(index=False))

    return(valjc)

def cluster_compactness(clust_stats,compactness,spatially_constrained):
    """
    Compactness statistics from pygeoda.spatial_validation

    Arguments:
    ----------
    clust_stats           : cluster cardinalities
    compactness           : compactness attribute from pygeoda.spatial_validation
                            area, perimeter and isoperimeteri quotient
    spatially_contrained  : flag for spatially_constrained from pygeoda.spatial_validation

    Returns:
    --------
    valcomp               : data frame with compactness statistics
    
    """

    if not(spatially_constrained):
        print("Error: Compactness is only applicable to spatially constrained clusters")
        return
    
    comp = compactness

    compt = []
    jj = 0

    for i in comp:
        nn = clust_stats['Cardinality'].iloc[jj]
        compt.append({
            'Label' : jj,
            'N' : nn,
            "Area" : i.area,
            "Perimeter" : i.perimeter,
            "IPQ" : i.isoperimeter_quotient
        })
        jj = jj + 1

    valcomp = pd.DataFrame(compt)

    print("Compactness")
    print(valcomp.to_string(index=False))

    return(valcomp)

def cluster_diameter(clust_stats,diameter,spatially_constrained):

    if not(spatially_constrained):
        print("Error: Diameter is only applicable to spatially constrained clusters")
        return

    diam = diameter

    diamt = []
    jj = 0
    for i in diam:
        nn = clust_stats['Cardinality'].iloc[jj]

        diamt.append({
            'Label' : jj,
            'N' : nn,
            "Steps" : i.steps,
            "Ratio" : i.ratio
        })
        jj = jj + 1

    valdiam = pd.DataFrame(diamt)

    print("Diameter")
    print(valdiam.to_string(index=False))

    return(valdiam)

def ensure_datasets(
    expected_file,
    folder_path="./datasets/",
    zip_url=None):

    """
    Ensures that the required dataset file exists locally.

    If the specified file is not found in the given folder path, this function downloads
    a zip file from the provided URL and extracts its contents to the folder path.

    Parameters
    ----------
    expected_file : str
        Relative path to the expected file (e.g., 'ceara/ceara.shp') inside the folder_path.
    folder_path : str, optional
        Local folder path where the datasets should be found or extracted to.
        Default is './datasets/'.
    zip_url : str, optional
        URL of the zip file containing all required datasets.
        Default is None.

    Raises
    ------
    RuntimeError
        If the download fails (e.g., response status is not 200).

    Notes
    -----
    This function assumes that the zip file contains nested folders with the required structure.
    """
    
    expected_path = os.path.join(folder_path, expected_file)

    if not os.path.exists(expected_path):
        print(f"'{expected_path}' not found. Downloading and extracting dataset...")
        if zip_url is None:
            if 'ceara' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/ceara.zip'
            elif 'chicago_2020_sdoh' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/Chi_CCA.zip'
            elif 'chi-sdoh' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/Chi-SDOH.zip'
            elif 'chicago_commpop' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/chicago_commpop.zip'
            elif 'italy_banks' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/italy_banks.zip'
            elif any(x in expected_path.lower() for x in ['liq_chicago', 'chicagoboundary']):
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/liquor.zip'
            elif 'spirals' in expected_path.lower():
                zip_url = 'https://raw.githubusercontent.com/lanselin/notebooks_for_spatial_clustering/4b54a0019aef4cc7ebb75526eb3113b05f34b2a3/datasets/spirals.zip'                                                          
            else:
                raise ValueError(f"Unidentified data: could not determine download URL for path '{expected_path}'")

        response = requests.get(zip_url)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(folder_path)
            print("Download and extraction completed.")
        else:
            raise RuntimeError("Failed to download the dataset zip file.")
    else:
        #print(f"'{expected_path}' already exists. No download needed.")
        pass
