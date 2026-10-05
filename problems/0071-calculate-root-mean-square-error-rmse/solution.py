
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	if y_true.shape != y_pred.shape or y_true.size == 0 or y_pred.size == 0:
		raise ValueError("")
	else:
		n = y_true.shape[0]
		rmse_res = np.sqrt(np.mean((y_true-y_pred)**2)) 
		return round(rmse_res,3)
