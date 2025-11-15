# Bio-Inspired Computation Coursework /25
#  ANN->forward pass 
# net takes 8 inputs and predicts 1 output (regression)

import argparse
import numpy as np
import pandas as pd


# ACTIVATIONS
def relu(z): 
    return np.maximum(0, z)

def tanh(z): 
    return np.tanh(z)

def identity(z): 
    return z   

def forward(X, weights, biases):
    A = X

    Z = A @ weights[0] + biases[0]
    A = relu(Z)

    Z = A @ weights[1] + biases[1]
    A = tanh(Z)

    Z = A @ weights[2] + biases[2]
    A = identity(Z)

    return A



# PARAMETER INITIAL
def init_params(layer_sizes, randomNo):
    weight, biases = [], []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        w_i = randomNo.normal(0.0, 0.1, size=(n_in, n_out)).astype(np.float64)
        b_i = np.zeros((n_out,), dtype=np.float64)

        weight.append(w_i)
        biases.append(b_i)

    return weight, biases


# PSO LATER(FLAT / UNFLAT)
def count_params(weight, biases):
    tot = 0
    for w_i, b_i in zip(weight, biases):
        tot += w_i.size + b_i.size
    return tot

def pack_params(weight, biases):
    flat = []
    for w_i, b_i in zip(weight, biases):
        flat.append(w_i.ravel())
        flat.append(b_i.ravel())
    return np.concatenate(flat)

def unpack_params(flat, shape_w, shape_b):
    weight, biases = [], []
    offset = 0
    for (rows, cols), (d,) in zip(shape_w, shape_b):
        size_w = rows * cols
        w_i = flat[offset:offset + size_w].reshape(rows, cols)
        offset += size_w
        b_i = flat[offset:offset + d]
        offset += d
        weight.append(w_i)
        biases.append(b_i)
    return weight, biases



# DATA LOADER
def load_concrete(csv_path, test_ratio=0.3, startVal=123):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 0:8].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)

    randomNo = np.random.default_rng(startVal)
    idx = randomNo.permutation(len(X))

    n_test = int(round(test_ratio * len(X)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, y_train, X_test, y_test


# METRIC
def mae(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


# ARGUMENTS
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", 
        type=str,
        default="Data/Concrete_Data_Yeh.csv"
    )
    
    
    parser.add_argument(
        "--layers", 
        type=int, 
        nargs="+",
        default=[8, 16, 8, 1]
    )


    parser.add_argument(
        "--startVal",
        type=int, 
        default=42
    )
    
    
    return parser.parse_args()


# MAIN()

def main():
    args = get_args()
    X_train, Y_train, X_test, Y_test = load_concrete(args.csv)
    randomNo = np.random.default_rng(args.startVal)
    weight, biases = init_params(args.layers, randomNo)


    tot_params = count_params(weight, biases)
    print(f"Total parameters: {tot_params}")
    flat = pack_params(weight, biases)
    flat = randomNo.normal(0.0, 0.2, size=flat.size)
    W_shapes = [w.shape for w in weight]
    B_shapes = [b.shape for b in biases]

    weight, biases = unpack_params(flat, W_shapes, B_shapes)
    Yhat = forward(X_test, weight, biases)
    test_mae = mae(Y_test, Yhat)


    print(f"{args.layers[0]} -> {args.layers[1]} (ReLU)")
    print(f"{args.layers[1]} -> {args.layers[2]} (tanh)")
    print(f"{args.layers[2]} -> {args.layers[3]} (identity)")

    print("\nTest MAE with random weights:", round(test_mae, 4))
