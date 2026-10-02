import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    if len(b) != len(a[0]):
        return -1
    return np.matmul(a, b).tolist()