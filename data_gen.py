from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

def generate_blob_data(n_samples=1000, n_features=5, center=10):
    X, y_true=make_blobs(n_samples=n_samples, n_features=n_features, centers=center, random_state=0)
    return X, y_true

def generate_kddcup99(n_samples=2000, n_components=20, random_state=0):
    
    kdd_data = fetch_openml("KDDCup99", version=1, as_frame=True, parser="liac-arff")
    
    df = kdd_data.frame.copy()

    df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    y = df["label"]
    X = df.drop("label", axis=1)

    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_numeric = X[numeric_cols].to_numpy()
    X_processed = np.hstack([X_numeric, X_encoded])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, y.to_numpy()


def load_fashion_mnist_pca(n_components=2):
    fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")

    X = fashion.data
    y_true = fashion.target
    np.random.seed(0)
    indices = np.random.choice(X.shape[0], size=1000, replace=False)
    X=X[indices]
    y_true=y_true[indices]
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    return X_scaled, y_true
if __name__=='__main__':
    kcc=load_fashion_mnist_pca()
    X, y_true=kcc
    print(X.shape)