#Bio-Inspired Computation Coursework/25
#Task 1: Basic feed-forward ANN(forward pass only)
#THe net takes 8 inputs and predicts 1 output(reps regression)
import argparse
import numpy as np  
import pandas as pd

# ----------------------
# Activation func.s
# ----------------------
#hidden layers: 
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
# Network helpers - Parameter init + forward pass
# ----------------------

def init_params(layer_sizes, randomNo):
    """
    Creates weight matrices and bias vectors for each layer.
    Sizes come from the list in layer_sizes, e.g. [8, 16, 8, 1].
    """
    #for each layer i:
    weight, biases = [], []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        #weight matrix: n_in x n_out
        n_out = layer_sizes[i + 1]
        #bias vector of shape i_bias: n_out
        # Simple small-normal init (works fine for this coursework)
        i_weight = randomNo.normal(0.0, 0.1, size=(n_in, n_out)).astype(np.float64)
        i_bias = np.zeros((n_out,), dtype=np.float64)
        weight.append(i_weight)
        biases.append(i_bias)
    return weight, biases


def forward(X, weight, biases, acts):
    """Simple formward pass through all layers.
    for each layer: Z = A @ weight+biases, then apply activation func.
    """
    #matrix multiply A @ i_weight (applies weight) +i_bias (adds biases)
    #repeat for all layers, use identity at output for regression
    A = X
    for i_weight, i_bias, name in zip(weight, biases, acts):
        Z = A @ i_weight + i_bias
        A = ACT[name](Z)
    return A

# ----------------------
# Flatten/unflatten params(for PSO later)
# ----------------------
#single vector flat contains all weights and biases -> flat
def count_params(weight, biases):
    tot = 0
    for i_weight, i_bias in zip(weight, biases):
        tot += i_weight.size + i_bias.size
    return tot


def pack_params(weight, biases):
    flat = []
    for i_weight, i_bias in zip (weight, biases):
        flat.append(i_weight.ravel())
        flat.append(i_bias.ravel())
    return np.concatenate(flat)


def unpack_params(flat, shape_w, shape_b):
    """Return (weight,biases) lists shaped from flat.
    W_shapes: list of (rows, cols) per layer
    B_shapes: list of (dim,) per layer
    """
    #puts flat vector back into weight/bias shapes. -> so PSO can update ANN
    weight, biases = [], []
    offset = 0
    for (rows, cols), (d,) in zip(shape_w, shape_b):
        size_w = rows * cols
        i_weight = flat[offset:offset + size_w].reshape(rows, cols)
        offset += size_w
        i_bias = flat[offset:offset + d]
        offset += d
        weight.append(i_weight)
        biases.append(i_bias)
    return weight, biases

# ----------------------
# Dataset loader + metric
# ----------------------

def load_concrete(csv_path, test_ratio=0.3, startVal=123):
    """Read CSV, first 8 columns->inputs, last column->targetVal.
    Data split into train/test then normalised using training-set mean/stddev.
    """
    #CANNOT split up the training and testing
    #avoids feats w/ different scales dominating
    #use train stats to avoid data leakage
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 0:8].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)

    randomNo = np.random.default_randomNo(startVal)
    n = len(X)
    idx = np.arange(n)
    randomNo.shuffle(idx)


    n_test = int(round(test_ratio * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # standardize feats
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, y_train, X_test, y_test

#avg absolute difference between true and predicted->for regression
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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="Data/Concrete_Data_Yeh.csv",
        help="Path to concrete CSV data.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 16, 8, 1],
        help="Layer sizes, including input and output.",
    )
    parser.add_argument(
        "--acts",
        type=str,
        nargs="+",
        default=["relu", "tanh", "identity"],
        help="Activation functions for each layer .",   
          #(len = len(layers)-1)
    )
    parser.add_argument(
        "--startVal",
        type=int,
        default=42,
        help="Random start value for weight init.",
    )
    return parser.parse_args()

def main():
    args = get_args()
    X_train, Y_train, X_test, Y_test = load_concrete(args.csv)
    randomNo = np.random.default_randomNo(args.startVal)
    weight, biases = init_params(args.layers, randomNo)
#(for PSO later)
    
    tot_params = count_params(weight, biases)
    print(f"Total parameters: {tot_params}")

    flat = pack_params(weight, biases)
    flat = randomNo.normal(0.0, 0.2, size=flat.size)

    W_shapes = [w.shape for w in W]
    B_shapes = [b.shape for b in B]
    weight, biases = unpack_params(flat, W_shapes, B_shapes)
    Yhat = forward(X_test, weight, biases, args.acts)
    test_mae = mae(Y_test, Yhat)

    print("Network:")
    for i in range(len(args.layers) - 1):
        print(f"  {args.layers[i]} -> {args.layers[i+1]}  act={args.acts[i]}")
    print(f"Test MAE with random weights: {test_mae:.4f}")

  
  #Treat flat as a particle position.")
  #unpack_params(flat) -> (weight,biaases)")
  #pred = forward(Xval,weight,biasesacts); fitness = MAE(yval, pred)  (minimize)")
    