"""One-vs-rest logistic regression used to train Frame Lab.

Runtime classification lives in classify.py. This module keeps the original
training math from the Graphic Simulation assignment.
"""

import copy
import json
import math
import sys

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0.0
    y = np.asarray(y).reshape(-1)
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        aux = np.where(f_wb_i > 1e-7, f_wb_i, 1e-7)
        aux2 = np.where((1 - f_wb_i) > 1e-7, (1 - f_wb_i), 1e-7)
        cost += -y[i] * np.log(aux) - (1 - y[i]) * np.log(aux2)

    cost = cost / m
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost


def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    m, n = X.shape
    y = np.asarray(y).reshape(-1)
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        dj_dw += err_i * X[i]
        dj_db += err_i

    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, r_lambda, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, r_lambda)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost_logistic_reg(X, y, w, b, r_lambda))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")

    return w, b, J_history


def y_change(y, cl):
    return [1 if value == cl else 0 for value in y]


def find_param(X, y):
    w_in = np.random.rand(X.shape[1])
    b_in = 0.5
    alph = 0.1
    r_lambda = 0.7
    iters = 1000

    y = np.asarray(y).reshape(-1)
    y_uniq = list(dict.fromkeys(y.tolist()))
    theta_list = []
    for class_id in y_uniq:
        y_tr = np.array(y_change(y, class_id))
        print(f"\n\nWe will find the weights for class: {class_id}")
        theta1, _, _ = gradient_descent(X, y_tr, w_in, b_in, alph, r_lambda, iters)
        theta_list.append(theta1)
    return theta_list


def predict(theta_list, X, y=None):
    """Predict a class id for each row using one-vs-rest argmax."""
    weights = np.array(theta_list, dtype=float)
    logits = np.dot(X, weights.T)
    return logits.argmax(axis=1) + 1


def extractFeature(img):
    from classify import extract_features
    return extract_features(img)


if __name__ == "__main__":
    from classify import classify_image

    if len(sys.argv) < 2:
        print("Usage: python imageClassifier.py <image_path>")
        sys.exit(1)
    print(json.dumps(classify_image(sys.argv[1]), indent=2))
