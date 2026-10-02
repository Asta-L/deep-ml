def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if not samples:
        return []
    import numpy as np
    vals, counts = np.unique(samples, return_counts = True)
    total_counts = len(samples)
    return [(int(val), float(count/total_counts)) for val, count in zip(vals,counts)]
    pass