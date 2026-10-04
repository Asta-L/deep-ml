def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    if not x:
        return []
    min_ = min(x)
    max_ = max(x)
    if max_ == min_:
        return [0] * len(x)
    
    return [round((a - min_)/(max_-min_),4) for a in x]
