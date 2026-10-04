import numpy as np
from collections import Counter
import math

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    classes = list(set(y_train))
    counts = Counter(y_train)

    if strategy == 'most_frequent':
        max_val = max(counts.values())
        best_key = min([k for k, v in counts.items() if v == max_val])
        result = [best_key]*n_test

    if strategy == 'constant':
        result = [constant]*n_test

    if strategy == 'uniform':
        sort = sorted(classes)
        k = len(classes)
        result = [sort[i%k] for i in range(n_test)]

    if strategy == 'stratified':
        allocations = {}
        fractions = {}
        for c in counts:
            ori = n_test*counts[c]/len(y_train)
            val = math.floor(ori)
            allocations[c] = val
            fractions[c] = ori - val

        remainder = n_test - sum(allocations.values())
        if remainder > 0:
            sorted_cls= sorted(classes, key = lambda c: (-fractions[c], c))
        for i in range(remainder):
            allocations[sorted_cls[i]] += 1

        result = []
        for c in classes:
            result.extend([c] * allocations[c])
            
    return result
            


        

        
