import numpy as np
import pandas as pd

x_train = pd.read_csv("train_in.csv", header=None).to_numpy()
y_train = pd.read_csv("train_out.csv", header=None).squeeze("columns").to_numpy()
x_test  = pd.read_csv("test_in.csv", header=None).to_numpy()
y_test  = pd.read_csv("test_out.csv", header=None).squeeze("columns").to_numpy()

# Compute class centers
centers = np.zeros((10, x_train.shape[1]))
for d in range(10):
    centers[d] = x_train[y_train == d].mean(axis=0)

# Classify One Point
def nearest_mean_predict(x, centers):
    # Computation of Distance
    dists = np.linalg.norm(centers - x, axis = 1)
    # Smallest Distance
    return np.argmin(dists)

# Classify Dataset
def nearest_mean_classify(z, centers):
    return np.array([nearest_mean_predict(x, centers) for x in z])

# Train & Test
y_train_pred = nearest_mean_classify(x_train, centers)
y_test_pred = nearest_mean_classify(x_test, centers)

train_acc = np.mean(y_train_pred == y_train) * 100
test_acc = np.mean(y_test_pred == y_test) * 100

print("Nearest Mean Classifier Accuracy: ")
print(f"  Train: {train_acc:.2f}%")
print(f"  Test:  {test_acc:.2f}%")

