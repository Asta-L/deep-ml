import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if not n_col:
		n_col = max(x)+1
	encoded = []
	for i in x:
		init_ = [0]*n_col
		init_[i] = 1
		encoded.append(init_)
	return encoded