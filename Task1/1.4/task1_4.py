from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x_train = pd.read_csv("train_in.csv", header=None).to_numpy()
y_train = pd.read_csv("train_out.csv", header=None).squeeze("columns").to_numpy()
x_test  = pd.read_csv("test_in.csv", header=None).to_numpy()
y_test  = pd.read_csv("test_out.csv", header=None).squeeze("columns").to_numpy()

# Neighbors k = 3 for not too much noise
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)

# Predictions
y_train_pred_knn =  knn.predict(x_train)
y_test_pred_knn = knn.predict(x_test)

# Accuaracy
train_acc_knn = (y_train_pred_knn == y_train).mean() * 100
test_acc_knn = (y_test_pred_knn == y_test). mean() * 100
print(f"KNN Classifier Accuracy (k=3)")
print(f"  Train: {train_acc_knn:.2f}%")
print(f"  Test:  {test_acc_knn:.2f}%")

# for test set
cm_knn = confusion_matrix(y_test, y_test_pred_knn, labels = range(10))
disp = ConfusionMatrixDisplay(confusion_matrix = cm_knn, display_labels = range(10))
disp.plot(cmap = "Blues", values_format = "d")
plt.title("Confusion Matrix - KNN Test set.)")
plt.show()