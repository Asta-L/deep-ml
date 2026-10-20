import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    # 1. Standardize the dataset (mean=0, std=1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_data = (data - mean) / np.where(std == 0, 1, std)

    # 2. Compute the covariance matrix (divide by n-1 for sample covariance, or use np.cov)
    # Using np.cov with rowvar=False is standard for samples in rows, features in columns
    cov_matrix = np.cov(std_data, rowvar=False)
    cov_matrix = np.atleast_2d(cov_matrix)

    # 3. Perform eigendecomposition using np.linalg.eigh for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 5. Select the top-k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    # 6. Apply Sign Convention:
    # For each column vector, find the first element with abs > 1e-10; if negative, flip sign.
    for col in range(top_k_eigenvectors.shape[1]):
        vec = top_k_eigenvectors[:, col]
        nonzero_idx = np.where(np.abs(vec) > 1e-10)[0]
        if len(nonzero_idx) > 0 and vec[nonzero_idx[0]] < 0:
            top_k_eigenvectors[:, col] *= -1

    # 7. Return rounded results to 4 decimal places
    return np.round(top_k_eigenvectors, 4)
