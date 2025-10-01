
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
x_train = pd.read_csv("train_in.csv", header=None).to_numpy()
y_train = pd.read_csv("train_out.csv", header=None).squeeze("columns").to_numpy()
x_test  = pd.read_csv("test_in.csv", header=None).to_numpy()
y_test  = pd.read_csv("test_out.csv", header=None).squeeze("columns").to_numpy()

# Bias feature (column of 1s)
T_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
T_test  = np.hstack([x_test,  np.ones((x_test.shape[0], 1))])

n_features = T_train.shape[1]  # 257
n_classes  = 10

def train_perceptron(lr, epochs=50):
    W = np.zeros((n_features, n_classes))
    train_accs, val_accs, mistakes_lists = [], [], []

    for epoch in range(epochs):
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
        val_acc   = (np.argmax(T_test  @ W, axis=1) == y_test).mean()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[lr={lr}] Epoch {epoch+1}, Mistakes: {mistakes}, "
              f"Train acc: {train_acc:.2f}, Test acc: {val_acc:.2f}")

    return train_accs, val_accs, mistakes_lists

# Run for different learning rates
learning_rates = [0.1, 0.01, 0.001]
results = {}

for lr in learning_rates:
    train_accs, val_accs, mistakes_lists = train_perceptron(lr)
    results[lr] = {
        "train": train_accs,
        "val": val_accs,
        "mistakes": mistakes_lists
    }

# --- Plot mistakes for all learning rates ---
plt.figure(figsize=(6,4))
for lr in learning_rates:
    plt.plot(results[lr]["mistakes"], label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Mistakes")
plt.title("Perceptron Mistakes per Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot validation accuracy for all learning rates ---
plt.figure(figsize=(6,4))
for lr in learning_rates:
    plt.plot(np.array(results[lr]["val"]) * 100, label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.show()

