import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	n, m = X.shape
	weights = np.zeros(m+1)
	X_b = np.hstack([np.ones((n,1)), X])
	losses = []
	for _ in range(iterations):
		z = np.dot(X_b, weights)
		sigmoid = 1/(1+np.exp(-z))
		gradient = np.dot(X_b.T, sigmoid - y)
		loss = -(np.dot(y, np.log(sigmoid)) + np.dot(1-y, np.log(1-sigmoid)))
		weights -= learning_rate * gradient
		losses.append(loss)

	return (weights, losses)
		