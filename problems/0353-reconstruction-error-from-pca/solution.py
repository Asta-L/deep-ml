import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    X_arr = np.array(X, dtype=float)
    
    # 1. Center the data by subtracting the mean of each feature
    mean = np.mean(X_arr, axis=0)
    X_centered = X_arr - mean
    
    # 2. Compute the population covariance matrix (divide by n)
    n_samples = X_arr.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    
    # 3. Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 4. Select top n_components eigenvectors
    W = eigenvectors[:, :n_components]
    
    # 5. Project centered data onto principal components
    Z = np.dot(X_centered, W)
    
    # 6. Reconstruct data back to the original space (adding mean back)
    X_reconstructed = np.dot(Z, W.T) + mean
    
    # 7. Compute Mean Squared Error (MSE)
    mse = float(np.mean((X_arr - X_reconstructed) ** 2))
    
    return round(mse, 4)
    pass