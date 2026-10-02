import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
import numpy as np

def gradient_direction_magnitude(gradient: list[float]) -> dict:
    # Compute L2 norm (magnitude) as a standard float
    magnitude = float(np.linalg.norm(gradient))
    
    # Handle zero vector edge case
    if magnitude == 0:
        zero_vec = [0.0] * len(gradient)
        return {
            'magnitude': 0.0,
            'direction': zero_vec,
            'descent_direction': zero_vec
        }
    
    # Compute unit directions
    direction = [x / magnitude for x in gradient]
    descent_direction = [-x / magnitude for x in gradient]
    
    return {
        'magnitude': magnitude,
        'direction': direction,
        'descent_direction': descent_direction
    }

	