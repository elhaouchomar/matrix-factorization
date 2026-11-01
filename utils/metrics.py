import numpy as np


# Calculate the difference between actual and predicted values
def _difference(y_true, y_pred):
    return (
        np.asarray(y_true, dtype=float)
        - np.asarray(y_pred, dtype=float)
    )


# Calculate Root Mean Squared Error
def rmse(y_true, y_pred):
    diff = _difference(y_true, y_pred)

    return float(np.sqrt(np.mean(diff ** 2)))


# Calculate Mean Absolute Error
def mae(y_true, y_pred):
    diff = _difference(y_true, y_pred)

    return float(np.mean(np.abs(diff)))