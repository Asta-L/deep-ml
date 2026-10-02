def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'column':
		means = [sum(column)/len(column) for column in zip(*matrix)]
	else:
		means = [sum(row)/len(row) for row in matrix]
	
	
	
	return means