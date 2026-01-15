
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.int64)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print("Training Random Forest on raw MNIST...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )

    start_time = time.time()
    rf_clf.fit(X_train, y_train)
    rf_train_time = time.time() - start_time

    y_pred = rf_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)

    print(f"Raw RF training time: {rf_train_time:.2f} seconds")
    print(f"Raw RF accuracy: {rf_accuracy:.4f}")

    print("Applying PCA (95% variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Training Random Forest on PCA-reduced data...")
    rf_pca_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )

    start_time = time.time()
    rf_pca_clf.fit(X_train_pca, y_train)
    rf_pca_train_time = time.time() - start_time

    y_pred_pca = rf_pca_clf.predict(X_test_pca)
    rf_pca_accuracy = accuracy_score(y_test, y_pred_pca)

    print(f"PCA RF training time: {rf_pca_train_time:.2f} seconds")
    print(f"PCA RF accuracy: {rf_pca_accuracy:.4f}")

    print("Running t-SNE visualization on a subset...")
    np.random.seed(42)
    indices = np.random.choice(len(X_train), 2000, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_subset)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_subset,
        cmap="tab10",
        s=10
    )
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of MNIST digits")
    plt.show()

if __name__ == "__main__":
    main()
