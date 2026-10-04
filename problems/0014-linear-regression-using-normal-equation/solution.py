import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X)
	y = np.array(y)
	XTX_inv = np.linalg.inv(np.dot(X.T,X))
	Xy = np.dot(X.T,y)
	theta = np.dot(XTX_inv,Xy)
	return theta