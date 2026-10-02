import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    # Convert input to numpy array
    arr = np.array(data)
    
    # Mode calculation
    vals, counts = np.unique(arr, return_counts=True)
    mode_val = vals[np.argmax(counts)]
    
    # Basic statistics
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    variance_val = float(np.var(arr))
    std_val = float(np.std(arr))
    
    # Percentiles
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    iqr = float(p75 - p25)
    
    return {
        'mean': mean_val,
        'median': median_val,
        'mode': float(mode_val) if isinstance(mode_val, (int, float, np.number)) else mode_val,
        'variance': variance_val,
        'standard_deviation': std_val,
        '25th_percentile': p25,
        '50th_percentile': p50,
        '75th_percentile': p75,
        'interquartile_range': iqr
    }