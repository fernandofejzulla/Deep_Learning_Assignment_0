import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

## Loading CSV files
#Image 1707
x_train = pd.read_csv("train_in.csv", header = None).to_numpy()
y_train = pd.read_csv("train_out.csv", header = None).squeeze("columns").to_numpy()

#Image 1000
x_test = pd.read_csv("train_in.csv", header = None).to_numpy()
y_test = pd.read_csv("train_out.csv", header = None).squeeze("columns").to_numpy()

#Printing Check
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)
#print("labels present:", np.unique(y_train))

#Allocate an array to store 10 class center (256-dim)
centers = np.zeros((10, x_train.shape[1]), dtype = float)

for d in range(10):
    class_rows = x_train[y_train == d]
    centers[d] = class_rows.mean(axis = 0)

fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
fig.suptitle("Mean Image (Class Center) per Digit", y = 1.05)

for d, ax in enumerate(axes.ravel()):
    ax.imshow(centers[d].reshape(16, 16))
    ax.set_title(str(d))
    ax.axis("off")

plt.tight_layout()
plt.show()

## ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
# Gram matrix (10x10)
G = centers @ centers.T
# Squared Norms (10x1)
n2 = np.sum(centers ** 2, axis = 1, keepdims = True)
dist_sq = n2 + n2.T - 2 * G
# Numerical Floor at 0
dist_sq = np.maximum(dist_sq, 0)
dist_matrix = np.sqrt(dist_sq)

np.set_printoptions(precision=2, suppress=True)
print(dist_matrix)

