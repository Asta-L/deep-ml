import numpy as np
# from sklearn.dummy import dummy_regressor
def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    if strategy == 'mean':
        p = np.mean(y_train)
    if strategy == 'median':
        p = np.median(y_train)
    if strategy == 'quantile':
        if not quantile or quantile <0 or quantile>1:
            raise ValueError("Need quantile in [0,1]")
        else:
            p = np.quantile(y_train, quantile)
    if strategy == 'constant':
        if not constant:
            raise ValueError("constant cannot be none")
        else:
            p = constant

    return [float(p)]*n_test
