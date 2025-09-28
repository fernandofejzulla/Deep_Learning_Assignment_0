import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

x_train = pd.read_csv("train_in.csv", header=None).to_numpy()
y_train = pd.read_csv("train_out.csv", header=None).squeeze("columns").to_numpy()
x_test  = pd.read_csv("test_in.csv", header=None).to_numpy()
y_test  = pd.read_csv("test_out.csv", header=None).squeeze("columns").to_numpy()

#PCA
pca = PCA(n_components = 2, random_state = 42)
x_pca = pca.fit_transform(x_train)

plt.figure(figsize = (6, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y_train, cmap = "tab10", s = 10)
plt.colorbar()
plt.title("2D PCA")
plt.show()

#t-SNE
tsne = TSNE(n_components = 2, random_state = 42, init = "pca", learning_rate = "auto")
x_tsne = tsne.fit_transform(x_train)

plt.figure(figsize = (6, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y_train, cmap = "tab10", s = 10)
plt.colorbar()
plt.title("2D TSNE")
plt.show()

#UMAP
umap = umap.UMAP(n_components = 2, random_state = 42)
x_umap = umap.fit_transform(x_train)

plt.figure(figsize = (6, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y_train, cmap = "tab10", s = 10)
plt.colorbar()
plt.title("2D UMAP")
plt.show()