import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    n,d = X.shape
    m = len(batch_indices)
    batch_X = X[batch_indices]
    pred = np.dot(batch_X,weights)
    residual = pred - y[batch_indices] + bias
    gradient_w = 2/m * np.dot(batch_X.T, residual)
    gradient_b = 2/m * np.sum(residual)
    weights -= lr*gradient_w
    bias -= lr*gradient_b
    return np.append(weights,bias)
