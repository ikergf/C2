import numpy as np

def heaviside(x, epsilon=1.0):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(x / epsilon))

def delta(x, epsilon=1.0):
    return epsilon / (np.pi * (epsilon**2 + x**2))

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    grad_i = np.zeros_like(image)  # vertical gradient
    grad_j = np.zeros_like(image)  # horizontal gradient

    # Forward difference for vertical gradient (axis 0)
    grad_i[:-1, :] = image[1:, :] - image[:-1, :]
    
    # Forward difference for horizontal gradient (axis 1)
    grad_j[:, :-1] = image[:, 1:] - image[:, :-1]

    return grad_i, grad_j

def im_bwd_gradient(image: np.ndarray):

    grad_i = np.zeros_like(image)
    grad_j = np.zeros_like(image)

    # Backward difference for vertical gradient (axis 0)
    grad_i[1:, :] = image[1:, :] - image[:-1, :]

    # Backward difference for horizontal gradient (axis 1)
    grad_j[:, 1:] = image[:, 1:] - image[:, :-1]

    return grad_i, grad_j