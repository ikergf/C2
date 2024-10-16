import numpy as np
from scipy.signal import fftconvolve, correlate2d

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    # grad_i = 0
    # grad_j = 0

    grad_i = np.zeros_like(image)  # vertical gradient
    grad_j = np.zeros_like(image)  # horizontal gradient

    # Forward difference for vertical gradient (axis 0)
    grad_i[:-1, :] = image[1:, :] - image[:-1, :]
    # grad_i[0, :] = image[0, :]  # Handle boundary
    
    # Forward difference for horizontal gradient (axis 1)
    grad_j[:, :-1] = image[:, 1:] - image[:, :-1]
    # grad_j[0, :] = image[:, 0]  # Handle boundary

    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    # div_i = 0
    # div_j = 0

    div_i = np.zeros_like(im1)
    div_j = np.zeros_like(im2)

    # Backward difference for vertical divergence (axis 0)
    div_i[1:, :] = im1[1:, :] - im1[:-1, :]
    # div_i[0, :] = im1[0, :]  # Handle boundary
    
    # Backward difference for horizontal divergence (axis 1)
    div_j[:, 1:] = im2[:, 1:] - im2[:, :-1]
    # div_j[:, 0] = im2[:, 0]  # Handle boundary

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

def get_translation(org_img: np.ndarray, dst_img: np.ndarray):
    # Perform correlation
    #correlation = correlate2d(dst_img, org_img, mode='full')
    correlation = fftconvolve(dst_img, np.flipud(np.fliplr(org_img)), mode='full')

    # Find max correlation value (best alignment)
    y_max, x_max = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Compute trnaslation
    shift_y = y_max - org_img.shape[0] + 1
    shift_x = x_max - org_img.shape[1] + 1
    
    return (shift_y, shift_x)