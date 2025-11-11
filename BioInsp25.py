#Bio-Inspired Computation Coursework/25
#8 inputs to 1 numeric target
#defines forward-only neural net
#evals with MAE
import argparse
import numpy as np  
import pandas as pd

# ----------------------
# Activation functions
# ----------------------
#hidden layers: identity(no squashing)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x
#dictionary of activations
ACT = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "identity": identity,
    "linear": identity,  # alias
}

# ----------------------
# Network helpers
# ----------------------

def init_params(layer_sizes, rng):
    """Return (W, B) lists shaped from flat theta.
W_shapes: list of (rows, cols) per layer
B_shapes: list of (dim,) per layer
"""
#for each layer i -> i+1
    W, B = [], []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        #weight matrix Wi: n_in x n_out
        n_out = layer_sizes[i + 1]
        #bias vector of shape bi: n_out
        # Simple small-normal init (works fine for this coursework)
        Wi = rng.normal(0.0, 0.1, size=(n_in, n_out)).astype(np.float64)
        bi = np.zeros((n_out,), dtype=np.float64)
        W.append(Wi)
        B.append(bi)
    return W, B


def forward(X, W, B, acts):
    """Forward pass through all layers.
    acts is a list of activation names, len == len(W)
    """
    #matrix multiply A @ Wi (applies weight) + bi (adds biases)
    #repeat for all layers, use identity at output for regression
    A = X
    for Wi, bi, name in zip(W, B, acts):
        Z = A @ Wi + bi
        A = ACT[name](Z)
    return A

# ----------------------
# Pack / unpack flat vector (for PSO later)
# ----------------------
#single vector theta contains all weights and biases -> theta
def count_params(W, B):
    total = 0
    for Wi, bi in zip(W, B):
        total += Wi.size + bi.size
    return total


def pack_params(W, B):
    parts = []
    for Wi, bi in zip(W, B):
        parts.append(Wi.ravel())
        parts.append(bi.ravel())
    return np.concatenate(parts)


def unpack_params(theta, W_shapes, B_shapes):
    """Return (W, B) lists shaped from flat theta.
    W_shapes: list of (rows, cols) per layer
    B_shapes: list of (dim,) per layer
    """
    W, B = [], []
    off = 0
    for (r, c), (d,) in zip(W_shapes, B_shapes):
        size_w = r * c
        Wi = theta[off:off + size_w].reshape(r, c)
        off += size_w
        bi = theta[off:off + d]
        off += d
        W.append(Wi)
        B.append(bi)
    return W, B

# ----------------------
# Data + metric
# ----------------------

def load_concrete(csv_path, test_ratio=0.3, seed=123):
    """Read CSV, first 8 columns are inputs, last column is target.
    Standardize inputs using train statistics only.
    """
    #avoids feats w/ different scales dominating
    #use train stats to avoid data leakage
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 0:8].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_ratio * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # standardize features
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, y_train, X_test, y_test

#avg absolute difference between true and predicted (for regression)
def mae(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

# ----------------------
# Main (demo with random weights)
# ----------------------
#1.parse CLI args
#2.check/load data
#3.spltit standardize features
#4.init network params weights/biases
#5.show param count + pack/unpack demo
#6.forward on test set + compute MAE

def main():
    parser = argparse.ArgumentParser(description="ANN (Task 1) — forward only")
    parser.add_argument("csv", help="Path to Concrete Compressive Strength CSV")
    parser.add_argument("--layers", nargs="*", type=int, default=[8, 16, 8, 1],
                        help="Layer sizes incl. input & output, e.g. 8 16 8 1")
    parser.add_argument("--acts", nargs="*", type=str, default=["relu", "tanh", "identity"],
                        help="Activations per layer (same length as layers-1)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    # Basic checks for this dataset
    if args.layers[0] != 8:
        raise SystemExit("Input layer must be 8 for this dataset.")
    if args.layers[-1] != 1:
        raise SystemExit("Output layer must be 1 for regression.")
    if len(args.acts) != len(args.layers) - 1:
        raise SystemExit("--acts length must equal len(--layers) - 1")
    for a in args.acts:
        if a not in ACT:
            raise SystemExit(f"Unknown activation: {a}")

    # Load data
    Xtr, Ytr, Xte, Yte = load_concrete(args.csv)

    # Init network params
    rng = np.random.default_rng(args.seed)
    W, B = init_params(args.layers, rng)

    # Show param count and demonstrate pack/unpack (for PSO later)
    n_params = count_params(W, B)
    print(f"Total parameters: {n_params}")

    theta = pack_params(W, B)
    # Randomize theta a bit to show round-trip
    theta = rng.normal(0.0, 0.2, size=theta.size)

    # Shapes for unpacking
    W_shapes = [w.shape for w in W]
    B_shapes = [b.shape for b in B]
    W, B = unpack_params(theta, W_shapes, B_shapes)

    # Forward on test set and compute MAE
    Yhat = forward(Xte, W, B, args.acts)
    test_mae = mae(Yte, Yhat)

    print("Network:")
    for i in range(len(args.layers) - 1):
        print(f"  {args.layers[i]} -> {args.layers[i+1]}  act={args.acts[i]}")
    print(f"Test MAE with random weights: {test_mae:.4f}")

    print("PSO use:")
    print("  • Treat theta (flat vector) as a particle position.")
    print("  • unpack_params(theta) -> (W, B)")
    print("  • pred = forward(Xval, W, B, acts); fitness = MAE(yval, pred)  (minimize)")
    
if __name__ == "__main__":
    main()
    