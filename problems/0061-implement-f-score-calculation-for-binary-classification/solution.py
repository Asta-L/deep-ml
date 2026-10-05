import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	recall = np.sum((y_pred == 1)&(y_true ==1))/np.sum(y_true==1)
	presision = np.sum((y_pred == 1)&(y_true ==1))/np.sum(y_pred==1)
	if (beta**2 * presision + recall)==0:
		return 0
	F_score = (1+beta**2)*recall*presision/(beta**2 * presision + recall)

	return round(float(F_score),3)