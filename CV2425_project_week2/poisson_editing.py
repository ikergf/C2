import numpy as np

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    # grad_i = 0
    # grad_j = 0

    grad_i = np.zeros_like(image)  # vertical gradient
    grad_j = np.zeros_like(image)  # horizontal gradient

    # Forward difference for vertical gradient (axis 0)
    grad_i[:-1, :] = image[1:, :] - image[:-1, :]
    grad_i[0, :] = image[0, :]  # Handle boundary
    
    # Forward difference for horizontal gradient (axis 1)
    grad_j[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_j[0, :] = image[:, 0]  # Handle boundary

    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    # div_i = 0
    # div_j = 0

    div_i = np.zeros_like(im1)
    div_j = np.zeros_like(im2)

    # Backward difference for vertical divergence (axis 0)
    div_i[1:, :] = im1[1:, :] - im1[:-1, :]
    div_i[0, :] = im1[0, :]  # Handle boundary
    
    # Backward difference for horizontal divergence (axis 1)
    div_j[:, 1:] = im2[:, 1:] - im2[:, :-1]
    div_j[:, 0] = im2[:, 0]  # Handle boundary

    return div_i + div_j

def composite_gradients(vi1: np.array, vj1: np.array, vi2: np.array, vj2: np.array, mask: np.array):
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    # CODE TO COMPLETE
    # vi = 0
    # vj = 0

    # vi1, vj1 = im_fwd_gradient(u1)
    # vi2, vj2 = im_fwd_gradient(u2)

    vi = mask * vi1 + (1 - mask) * vi2
    vj = mask * vj1 + (1 - mask) * vj2
    return vi, vj

def poisson_linear_operator(u: np.array, beta: np.array):
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
    # Au = 0

    # grad_i, grad_j = im_fwd_gradient(u)
    # beta_grad_i = beta * grad_i
    # beta_grad_j = beta * grad_j
    
    # Au = im_bwd_divergence(beta_grad_i, beta_grad_j)
    # CODE TO COMPLETE
    # Au = Au
    return 

def get_translation(original_img: np.ndarray, translated_img: np.ndarray, *part: str):

    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    # The following shifts are hard coded:
    if part[0] == "eyes":
        return (24, 8)
    elif part[0] == "mouth":
        return (33, 7)
    else:
        return (0, 0)

    # Here on could determine the shift vector programmatically,
    # given an original image/mask and its translated version.
    # Idea: using maximal cross-correlation (e.g., scipy.signal.correlate2d), or similar.