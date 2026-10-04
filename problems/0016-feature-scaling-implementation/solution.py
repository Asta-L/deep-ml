import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	
	mean = data.mean(axis = 0)
	std = data.std(axis = 0)
	standardized_data = (data - mean)/std
	min_ = data.min(axis = 0)
	max_ = data.max(axis = 0)
	normalized_data = (data - min_)/(max_ - min_)
	

	return standardized_data, normalized_data