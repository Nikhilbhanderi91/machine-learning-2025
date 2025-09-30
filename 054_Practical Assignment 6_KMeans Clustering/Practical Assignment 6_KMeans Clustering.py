# 1. Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 2. Importing the dataset from CSV
dataset = pd.read_csv("/content/E-commerce Customer Behavior - Sheet1.csv")
dataset.head()

# 3. Select numeric features for clustering
features = ['Total Spend', 'Items Purchased', 'Days Since Last Purchase', 'Average Rating']
X = np.array(dataset[features])

# 4. Function to calculate Euclidean distance
def calculate_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 5. Function to find the nearest centroid
def assign_clusters(centroids, X):
    assigned_cluster = []
    for i in X:
        distances = [calculate_distance(i, c) for c in centroids]
        assigned_cluster.append(np.argmin(distances))
    return assigned_cluster

# 6. Function to update centroids
def update_centroids(clusters, X):
    new_centroids = []
    df_temp = pd.concat([pd.DataFrame(X), pd.Series(clusters, name='cluster')], axis=1)
    for c in set(df_temp['cluster']):
        current_cluster = df_temp[df_temp['cluster'] == c][df_temp.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return np.array(new_centroids)

# 7. Initialize centroids randomly (choose k=3)
k = 3
init_indices = random.sample(range(len(X)), k)
centroids = np.array([X[i] for i in init_indices])
print("Initial Centroids:\n", centroids)

# 8. Train the model (Iterative updates)
epochs = 10
for i in range(epochs):
    clusters = assign_clusters(centroids, X)
    centroids = update_centroids(clusters, X)
    
    # Visualize first and last iteration
    if i == 0 or i == epochs - 1:
        plt.figure(figsize=(10,7))
        plt.scatter(X[:,0], X[:,1], c=clusters, alpha=0.3)
        plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='X', s=200)
        plt.title(f'K-Means Clustering - Iteration {i+1}')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.show()

# 9. Add cluster info to DataFrame
dataset['Cluster'] = clusters

# 10. Print cluster statistics
for i in range(k):
    print(f"\nCluster {i} Summary:")
    print(dataset[dataset['Cluster']==i][features].describe())