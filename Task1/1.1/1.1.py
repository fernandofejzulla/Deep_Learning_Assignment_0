import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# 1. Load data
train_in = pd.read_csv("train_in.csv", header=None).values
train_out = pd.read_csv("train_out.csv", header=None).values.ravel()
test_in = pd.read_csv("test_in.csv", header=None).values
test_out = pd.read_csv("test_out.csv", header=None).values.ravel()

# 2. Compute class centers
centers = np.array([train_in[train_out == d].mean(axis=0) for d in range(10)])

# Distances between centers
dist_matrix = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        dist_matrix[i,j] = np.linalg.norm(centers[i]-centers[j])

print("Distance matrix:\n", dist_matrix)

# 3. Dimensionality reduction
pca = PCA(n_components=2).fit_transform(train_in)
tsne = TSNE(n_components=2, random_state=42).fit_transform(train_in)
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(train_in)

fig, axes = plt.subplots(1,3, figsize=(15,5))
for ax, emb, title in zip(axes, [pca, tsne, umap_2d], ["PCA", "t-SNE", "UMAP"]):
    scatter = ax.scatter(emb[:,0], emb[:,1], c=train_out, cmap="tab10", s=10)
    ax.set_title(title)
plt.show()

# 4. Nearest Mean Classifier
def nearest_mean_classifier(X, centers):
    preds = []
    for x in X:
        dists = np.linalg.norm(centers - x, axis=1)
        preds.append(np.argmin(dists))
    return np.array(preds)

train_preds = nearest_mean_classifier(train_in, centers)
test_preds = nearest_mean_classifier(test_in, centers)

print("Nearest Mean Train Accuracy:", (train_preds == train_out).mean())
print("Nearest Mean Test Accuracy:", (test_preds == test_out).mean())

ConfusionMatrixDisplay(confusion_matrix(test_out, test_preds)).plot()
plt.title("Nearest Mean Confusion Matrix (Test)")
plt.show()

# 5. KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # try k=3,5
knn.fit(train_in, train_out)
train_knn_preds = knn.predict(train_in)
test_knn_preds = knn.predict(test_in)

print("KNN Train Accuracy:", (train_knn_preds == train_out).mean())
print("KNN Test Accuracy:", (test_knn_preds == test_out).mean())

ConfusionMatrixDisplay(confusion_matrix(test_out, test_knn_preds)).plot()
plt.title("KNN Confusion Matrix (Test)")
plt.show()
