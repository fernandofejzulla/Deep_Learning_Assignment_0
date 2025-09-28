import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = pd.read_csv("train_in.csv", header=None).to_numpy()
y_train = pd.read_csv("train_out.csv", header=None).squeeze("columns").to_numpy()
x_test  = pd.read_csv("test_in.csv", header=None).to_numpy()
y_test  = pd.read_csv("test_out.csv", header=None).squeeze("columns").to_numpy()

# Bias feature (column of 1s)
T_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
T_test  = np.hstack([x_test,  np.ones((x_test.shape[0], 1))])

W = np.zeros((T_train.shape[1], 10))  # 257 x 10

n_features = T_train.shape[1]  # 257
n_classes  = 10
lr = 0.1

# Track Metrtics
train_accs, test_accs, mistakes_lists = [], [], []

# Training
for epoch in range(20):
    mistakes = 0
    for i in range(T_train.shape[0]):
        x = T_train[i, :]
        y = y_train[i]
        y_hat = np.argmax(W.T @ x)
        if y_hat != y:
            W[:, y]     += lr * x
            W[:, y_hat] -= lr * x
            mistakes += 1
    
    mistakes_lists.append(mistakes)
    train_acc = (np.argmax(T_train @ W, axis=1) == y_train).mean()
    test_acc  = (np.argmax(T_test  @ W, axis=1) == y_test ).mean()
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch {epoch+1}, Mistakes: {mistakes}, "
          f"Train acc: {train_acc:.2f}, Test acc: {test_acc:.2f}")

    if mistakes == 0:
        break

# Final Accuracy Type
print(f"\nFinal Train Accuracy: {train_accs[-1]*100:.2f}%")
print(f"Final Test Accuracy: {test_accs[-1]*100:.2f}%")

# Plot Accuracy
plt.figure(figsize=(6,4))
plt.plot(np.array(train_accs) * 100, label = "Train")
plt.plot(np.array(test_accs) * 100, label = "Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Perception Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Mistakes ---
plt.figure(figsize=(6,4))
plt.plot(mistakes_lists)
plt.xlabel("Epoch") 
plt.ylabel("Mistakes")
plt.title("Perceptron Mistakes per Epoch")
plt.tight_layout()
plt.show()