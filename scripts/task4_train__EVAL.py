import numpy as np
import pandas as pd

# import ANN helpers from Task 1
from task1_ANN import (
    load_concrete,
    init_params,
    forward,
    mae,
    pack_params,
    unpack_params,
    get_args
)

#attempted gradient descent though not reliable keeps breaking
#training was adjusted  with randmo search to find better weights
#this similarly mimicks PSO characteristics without verlocity
def train_ANN(
    csv_path,
    layers=[8, 16, 8, 1],
    activate=["relu", "tanh", "identity"],
    startVal=0,
    steps=1000
):
    """Loads the data, splits into train/test, builds the ANN,
    and evaluates mean absolute error using random weights.
    """
    #start with rng and rand no.s
    #then flattened
    X_train, Y_train, X_test, Y_test = load_concrete(csv_path)
    randomNo = np.random.default_rng(startVal)
    weight, biases = init_params(layers, randomNo)
    flat = pack_params(weight, biases)

    #stored for later unflattening
    W_shapes = [w.shape for w in weight]
    B_shapes = [b.shape for b in biases]

    w0, b0 = unpack_params(flat, W_shapes, B_shapes)
    pred0 = forward(X_train, w0, b0)
    bestError = mae(Y_train, pred0)
    bestFlat = flat.copy()

    #loop to make small change to the weight
    #rebuild ANN
    for i in range(steps):
        trial = bestFlat + randomNo.normal(0.0, 0.05, size=bestFlat.size)
        weight_y, bias_y = unpack_params(trial, W_shapes, B_shapes)
        #this line rebuilds ANN with the trial weights
        pred = forward(X_train, weight_y, bias_y)
        error = mae(Y_train, pred)
        #checks results and keeps best weights
        if error < bestError:
            bestError = error
            bestFlat = trial.copy()
            #print every 200 runs to break up output but can change
        if i % 200 == 0:
            print("run", i, "MAE:", bestError)

    weight_y, bias_y = unpack_params(bestFlat, W_shapes, B_shapes)
    testPred = forward(X_test, weight_y, bias_y)
    testError = mae(Y_test, testPred)

    print("Final Test MAE:", testError)
    print("Final Train MAE:", bestError)
    return bestError, testError


if __name__ == "__main__":
    args = get_args()
    train_ANN(args.csv)

