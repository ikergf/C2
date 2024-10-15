import cv2
import numpy as np
import poisson_editing
from scipy.sparse import linalg
import inpainting
from scipy.sparse.linalg import cg  # Conjugate Gradient solver
from utils import *


from dataclasses import dataclass

@dataclass
class Parameters:
    hi: float
    hj: float

case = 'lena'

if case == 'lena':
    # Load images
    src = cv2.imread('images/lena/girl.png')
    dst = cv2.imread('images/lena/lena.png')

else:
    # For Mona Lisa and Ginevra:
    src = cv2.imread('images/monalisa/ginevra.png')
    dst = cv2.imread('images/monalisa/lisa.png')

min_val = np.min(src)
max_val = np.max(src)
im = (src.astype('float') - min_val)
src = im / max_val

min_val = np.min(dst)
max_val = np.max(dst)
im = (dst.astype('float') - min_val)
dst = im / max_val

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
# cv2.imshow('Source image', src); cv2.waitKey(0)
# cv2.imshow('Destination image', dst); cv2.waitKey(0)

if case == 'lena':
    # Load masks for eye swapping
    src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
    dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
    # cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
    # cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

    # Load masks for mouth swapping
    src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
    dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
    # cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
    # cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

    # Get the translation vectors (hard coded)
    t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
    t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

    # Shift the source image and masks based on the translations
    src_shifted_eyes, mask_shifted_eyes = shift_image(src, src_mask_eyes, t_eyes)
    src_shifted_mouth, mask_shifted_mouth = shift_image(src, src_mask_mouth, t_mouth)

    src_combined, mask = combine_sources_with_masks([src_shifted_eyes, src_shifted_mouth], dst, [mask_shifted_eyes, mask_shifted_mouth])

    mask_shifted_eyes = (mask_shifted_eyes > 128).astype('float')
    mask_shifted_mouth = (mask_shifted_mouth > 128).astype('float')


else:
    mask = cv2.imread('images/monalisa/mask.png', cv2.IMREAD_COLOR)
    print(src.shape)

    src_combined, mask = combine_sources_with_masks([src], dst, [mask])
    

# Cut out the relevant parts from the source image and shift them into the right position
# CODE TO COMPLETE

# Blend with the original (destination) image
# CODE TO COMPLETE
# mask = np.zeros_like(dst)
# u_comb = np.zeros_like(dst) # combined image

param = Parameters(0, 0)
param.hi = 1 / (ni-1)
param.hj = 1 / (nj-1)

# Combine masks for eyes and mouth
mask = (mask > 128).astype('float')

# Normalize values into [0,1]
# cv2.imshow('Final result of Poisson blending', im); cv2.waitKey(0)

# Initialize final blended image
u_comb = np.copy(dst)

for channel in range(3):

    # u1 = src_combined[:, :, channel]
    # f = dst[:, :, channel]

    m = mask[:, :, channel]
    u = src_combined[:, :, channel]
    f = dst[:, :, channel]
    if case == 'lena':
        u1 = src_shifted_eyes[:, :, channel]
        u2 = src_shifted_mouth[:, :, channel]

        gi_u1, gj_u1 = poisson_editing.im_fwd_gradient(u1)
        gi_u2, gj_u2 = poisson_editing.im_fwd_gradient(u2)
        gi_f, gj_f = poisson_editing.im_fwd_gradient(f)

        vi_aux, vj_aux = poisson_editing.composite_gradients(gi_u1, gj_u1, gi_f, gj_f, mask_shifted_eyes[:,:, channel])
        vi_aux2, vj_aux2 = poisson_editing.composite_gradients(gi_u2, gj_u2, vi_aux, vj_aux, mask_shifted_mouth[:,:, channel])

    else:
        u1 = src[:, :, channel]

        gi_u1, gj_u1 = poisson_editing.im_fwd_gradient(u1)
        gi_f, gj_f = poisson_editing.im_fwd_gradient(f)

        vi_aux2, vj_aux2 = poisson_editing.composite_gradients(gi_u1, gj_u1, gi_f, gj_f, mask[:,:, channel])


    beta_0 = 2.0   # TRY CHANGING
    beta = beta_0 * (1 - m)


    b = poisson_editing.im_bwd_divergence(vi_aux2, vj_aux2)
    
    x = inpainting.laplace_equation(u, m, b, beta, f, param)

    u_comb[:, :, channel] = x
    

cv2.imshow('Final result of Poisson blending', u_comb)
cv2.waitKey(0)