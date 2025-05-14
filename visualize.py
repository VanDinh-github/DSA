import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from DyDBSCAN import DynamicDBSCAN
from evaluator import ClusteringEvaluator
from data_gen import load_fashion_mnist_pca, generate_kddcup99
from sklearn.manifold import TSNE
from tuning import parameter_grid_search
from sklearn.preprocessing import LabelEncoder
import collections
# X, y_true = generate_kddcup99(n_samples=2000)

def visualize_clusters(X, min_p = 5, t = 10, eps = 1.5):
    # Create a scatter plot of the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title(f"DBSCAN Clustering (min_samples={min_p}, t={t}, eps={eps})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster Label")
    plt.show()
    
def visualize_clusters_states(X, min_p=5, t=10, eps=1.5):

    # Reduce to 2D using t-SNE for visualization
    print("Fitting t-SNE for 2D projection...")
    X_vis = TSNE(n_components=2, perplexity=30, init='random', learning_rate='auto', random_state=42).fit_transform(X)
    # print('Data:', X_vis)
    # Hyperparameters
    eps_list = [0.75, 1, 1.25, 1.5, 2, 3, 4, 5]
    min_samples_list = [5, 10, 15]
    t_list = [5, 10, 15]
    db = DynamicDBSCAN(int(min_p), int(t), float(eps), X)


    # cluster_labels_over_time = []
    def normalize_labels(labels):
        le = LabelEncoder()
        return le.fit_transform(labels)  # maps to [0, 1, ..., n_clusters - 1]

    import imageio
    from io import BytesIO

    # Re-initialize label list
    cluster_labels = [-1] * len(X)
    # cluster_labels_over_time = []

    frames = []  # Store in-memory frames

    print("Creating clustering visualization...")

    for i in range(len(X)):
        label_i = db.get_cluster(i)
        print(f"Point {i}: Cluster {label_i}")
        cluster_labels[i] = label_i
        # print(f"Point {i} assigned to cluster {label_i}")
        # cluster_labels_over_time.append(cluster_labels.copy())
        # print(f"Cluster labels: {cluster_labels}")
        normalize = normalize_labels(cluster_labels)
        # Create plot for current state
        # fig, ax = plt.subplots(figsize=(8, 6))
        if i%10 == 0 or (i in range(len(X)-5, len(X))):
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=normalize, cmap='tab20')
            ax.set_title(f"Dynamic DBSCAN Clustering - Step {i+1}")
            ax.axis('off')  # Optional: hide axis for cleaner visuals

            # Save plot to in-memory buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            if i == len(X) - 1:
                plt.savefig('result.png', format='png')

            buf.close()
            plt.close(fig)

    # Create final GIF
    imageio.mimsave("dynamic_dbscan.gif", frames, duration=0.2)
    print("GIF saved as dynamic_dbscan.gif")

if __name__ == '__main__':
    # X, y_true = load_fashion_mnist_pca()
    # X, y_true = generate_blob_data(n_samples=1000, n_features=5, center=10)
    X, y_true = generate_kddcup99(n_samples=2000)
    # visualize_clusters(X, y_true)
    visualize_clusters_states(X)
    # parameter_grid_search(X, y_true)