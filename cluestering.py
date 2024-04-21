import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from ast import literal_eval

def cluster(df, column_name, n_clusters=4):
    df[column_name] = df[column_name].apply(literal_eval).apply(np.array)
    matrix = np.vstack(df[column_name].values)

    kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    df['Cluster'] = kmeans.labels_

    centroids = kmeans.cluster_centers_
    centroid_dict = {}
    for label in range(n_clusters):
        # Mask to select only data points from the current cluster
        cluster_mask = (df['Cluster'] == label)
        cluster_data = matrix[cluster_mask]
        # Calculate distances from each point in the cluster to the centroid
        distances = np.linalg.norm(cluster_data - centroids[label], axis=1)
        # Get indices of the 10 minimum distances
        min_indices = np.argsort(distances)[:10]
        # Find the corresponding DataFrame rows
        closest_rows = df[cluster_mask].iloc[min_indices]
        centroid_dict[label] = closest_rows

    return df, centroid_dict

def optimal_cluster(df, column_name, method='silhouette', max_clusters=15):
    df[column_name] = df[column_name].apply(literal_eval).apply(np.array)
    matrix = np.vstack(df[column_name].values)

    wcss = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(matrix)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(matrix, labels))

    if method == 'elbow':
        kinks = np.diff(np.diff(wcss))
        n_clusters = np.argmax(kinks) + 2  # +2 because kinks index is offset by 1 from the range starting at 2 clusters
    elif method == 'silhouette':
        n_clusters = np.argmax(silhouette_scores) + 2  # +2 to adjust index to match the number of clusters

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    df['Cluster'] = kmeans.labels_

    centroids = kmeans.cluster_centers_
    centroid_dict = {}
    for label in range(n_clusters):
        # Mask to select only data points from the current cluster
        cluster_mask = (df['Cluster'] == label)
        cluster_data = matrix[cluster_mask]
        # Calculate distances from each point in the cluster to the centroid
        distances = np.linalg.norm(cluster_data - centroids[label], axis=1)
        # Get indices of the 10 minimum distances
        min_indices = np.argsort(distances)[:10]
        # Find the corresponding DataFrame rows
        closest_rows = df[cluster_mask].iloc[min_indices]
        centroid_dict[label] = closest_rows

    return df, centroid_dict
