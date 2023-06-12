"""
A program that trains a classification model to detect Breast Cancer

Wolberg,WIlliam. (1992). Breast Cancer Wisconsin (Original). 
UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z.

Attribute Information:

1. Sample code number:            id number
2. Clump Thickness:               1 - 10
3. Uniformity of Cell Size:       1 - 10
4. Uniformity of Cell Shape:      1 - 10
5. Marginal Adhesion:             1 - 10
6. Single Epithelial Cell Size:   1 - 10
7. Bare Nuclei:                   1 - 10
8. Bland Chromatin:               1 - 10
9. Normal Nucleoli:               1 - 10
10. Mitoses:                      1 - 10
11. Class:                        (2 for benign, 4 for malignant)


Class distribution: 357 benign, 212 malignant
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
N_ITERATIONS_MAX = 5000
LR = 0.085

def sigmoid(x):
    return 1 / (1 + (math.e ** -x))


def predict(x_i, w, b, scale):
    return sigmoid(np.dot(w, x_i) + b) > scale


# Cost
def cost(x, y, w, b):
    total_cost = 0.0
    m = x.shape[0]

    for i in range(m):
        prediction = sigmoid(np.dot(x[i], w) + b)
        total_cost += y[i] * math.log(prediction) + (1 - y[i]) * math.log(1 - prediction)
    return -total_cost / m


def load_data(filepath):
    x_values = []
    y_values = []

    with open(filepath) as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            line = line.split(',')

            x_values.append([int(value) for value in line[1:10:1]])
            y_values.append(line[10] == '4')
    return x_values, y_values


def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dw = np.zeros((n,))  # (n,)
    db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dw[j] = dw[j] + err_i * X[i, j]  # scalar
        db = db + err_i
    dw = dw / m  # (n,)
    db = db / m  # scalar

    return db, dw


# Trained data
x_t, y_t = load_data("data\cleaned_data.csv")
x_train = np.array(x_t[0:400:1])
y_train = np.array(y_t[0:400:1])

x_test = np.array(x_t[400:])
y_test = np.array(y_t[400:])

# Weights init
weights = np.zeros(x_train.shape[1])
bias = 0.

# Gradient descent
n_epoch = 0
calculated_cost = 0.
cost_history = []

while n_epoch < N_ITERATION_MAX:
    t_w = np.zeros(x_train.shape[1])
    loss_w = 0
    loss_b = 0

    # Calculate the gradient and update the parameters
    dj_db, dj_dw = compute_gradient_logistic(x_train, y_train, weights, bias)

    # Update Parameters using w, b, alpha and gradient
    weights -= LR * dj_dw
    bias -= LR * dj_db

    calculated_cost = cost(x_train, y_train, weights, bias)
    cost_history.append(calculated_cost)
    n_epoch += 1

    print("Epoch: " + str(n_epoch) + " | Calculated cost: " + str(calculated_cost))

n_trials = x_test.shape[0]
n_success = 0

n_malignant = 0
n_malignant_detected = 0
n_false_malignant = 0

for i in range(n_trials):
    if y_test[i]:
        n_malignant += 1
    if predict(x_test[i], weights, bias, 0.5) == y_test[i]:
        n_success += 1
    if predict(x_test[i], weights, bias, 0.25):
        if y_test[i]:
            n_malignant_detected += 1
        else:
            n_false_malignant += 1

success_rate = n_success/n_trials
print("Success rate: " + str(success_rate))
print("Final model parameters: " + str(weights) + " + " + str(bias) + '\n')

detection_rate = n_malignant_detected/n_malignant
print("Detection rate of with 25% assurance requested : " + str(detection_rate))
print("Within those, " + str(n_false_malignant) + " were benign.")

plt.plot(cost_history)
plt.show()
