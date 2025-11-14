import numpy as np
import pandas as pd

# import ANN helpers from Task 1
from task1_ANN import (
    load_concrete,
    init_params,
    forward,
    mae,
)


def train_ANN(
    csv_path="Data/Concrete_Data_Yeh.csv",
    layers=[8, 16, 8, 1],
    activate=["relu", "tanh", "identity"],
    startVal=0,
):
    """Loads the data, splits into train/test, builds the ANN,
    and evaluates mean absolute error using random weights.
    """

    print("ANN Eval (Random Weights)")
    X_train, Y_train, X_test, Y_test = load_concrete(csv_path)

    # Init ANN weights/biases
    randomNo = np.random.default_randomNo(startVal)
    W, B = init_params(layers, randomNo)
    preds = forward(X_test, W, B, activate)

    # Regression metric
    error = mae(Y_test, preds)
    print("Network architecture:")
    for i in range(len(layers) - 1):
        print(f"  {layers[i]} → {layers[i+1]}   act={activate[i]}")

    print(f"Test MAE (random weights): {error:.4f}")

    return error

if __name__ == "__main__":
    train_ANN()
