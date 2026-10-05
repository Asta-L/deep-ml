
from collections import Counter
import numpy as np
def confusion_matrix(data):
	# Implement the function here
	matrix = np.array(data)
	y_true = matrix[:,0]
	y_pred = matrix[:,1]
	TP = np.sum((y_pred==1)&(y_true==1))
	FP = np.sum((y_pred==1)&(y_true==0))
	TN = np.sum((y_pred==0)&(y_true==0))
	FN = np.sum((y_pred==0)&(y_true==1))
	return [[TP, FN], [FP, TN]]
