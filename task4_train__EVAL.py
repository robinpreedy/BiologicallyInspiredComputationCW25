import numpy as np
import pandas as pd
import task1_ANN import init_params, forward

def load_concrete(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)
    return X, y

