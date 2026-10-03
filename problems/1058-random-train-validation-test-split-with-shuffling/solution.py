import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n, d = data.shape
    random_row_index = np.random.default_rng(seed).permutation(n)
    shuffled_data = data[random_row_index]
    train_ind_end = int(train_frac * n)
    validation_ind_end = int(validation_frac*n) + train_ind_end
    return [shuffled_data[:train_ind_end], shuffled_data[train_ind_end:validation_ind_end],shuffled_data[validation_ind_end:]]
    