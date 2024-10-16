import cv2
import numpy as np
import poisson_editing
import inpainting
from utils import *
from dataclasses import dataclass
from create_custom_mask import create_custom_mask

@dataclass
class Parameters:
    hi: float
    hj: float

case = 'monalisa'

if case == 'lena':
    # Load images
    src = cv2.imread('images/lena/girl.png')
    dst = cv2.imread('images/lena/lena.png')
elif case == 'monalisa':
    # For Mona Lisa and Ginevra:
    src = cv2.imread('images/monalisa/ginevra.png')
    dst = cv2.imread('images/monalisa/lisa.png')
elif case == 'monalisa_custom':
    # For Mona Lisa and Ginevra:
    src = cv2.imread('images/monalisa/ginevra.png')
    dst = cv2.imread('images/monalisa/lisa.png')
elif case == 'lena_custom':
    src = cv2.imread('images/lena/girl.png')
    dst = cv2.imread('images/lena/lena.png')
elif case == 'madrid':
    src = cv2.imread('images/madrid/madrid1.jfif')
    dst = cv2.imread('images/madrid/madrid2.jpg')
elif case == 'dolphin':
    src = cv2.imread('images/dolphin/dolphin1.jpg')
    dst = cv2.imread('images/dolphin/dolphin2.jpg')
else:
    pass


# Normalization
min_val = np.min(src)
max_val = np.max(src)
im = (src.astype('float') - min_val)
src = im / max_val

min_val = np.min(dst)
max_val = np.max(dst)
im = (dst.astype('float') - min_val)
dst = im / max_val

padding = False

if src.shape != dst.shape:
    org_shape = dst.shape[:2]
    src, dst = match_image_sizes(src, dst)
    padding = True

src_masks = []
dst_masks = []

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
#cv2.imshow('Source image', src); cv2.waitKey(0)
#cv2.imshow('Destination image', dst); cv2.waitKey(0)

if case == 'lena':
    # Load masks for eye swapping
    src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_GRAYSCALE).astype('float')
    dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_GRAYSCALE).astype('float')

    cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
    cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

    src_masks.append(src_mask_eyes)
    dst_masks.append(dst_mask_eyes)

    # Load masks for mouth swapping
    src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_GRAYSCALE).astype('float')
    dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_GRAYSCALE).astype('float')

    cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
    cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

    src_masks.append(src_mask_mouth)
    dst_masks.append(dst_mask_mouth)

elif case == 'monalisa':
    # Load masks for face swapping
    mask = cv2.imread('images/monalisa/mask.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Eyes source mask', mask); cv2.waitKey(0)

    src_masks.append(mask)
    dst_masks.append(mask)
else:
    src_mask, dst_mask = create_custom_mask(src, dst)

    src_masks.append(src_mask)
    dst_masks.append(dst_mask)

translations = []
# Get the translation vectors
for src_m, dst_m in zip(src_masks, dst_masks):
    translations.append(poisson_editing.get_translation(src_m, dst_m))

src_shifted = []
mask_shifted = []
# Shift the source image and masks based on the translations
for src_m, tr in zip(src_masks, translations):
    src_s, mask_s = shift_image(src, src_m, tr)
    src_shifted.append(src_s)
    mask_shifted.append(mask_s)

src_combined, mask = combine_sources_with_masks(src_shifted, dst, mask_shifted)

mask_shifted_final = []

for i in mask_shifted:
    mask_shifted_final.append((i > 128).astype('float'))

param = Parameters(0, 0)
param.hi = 1 / (ni-1)
param.hj = 1 / (nj-1)

# Combine masks for eyes and mouth
mask = (mask > 128).astype('float')

# Initialize final blended image
u_comb = np.copy(dst)

for channel in range(3):

    m = mask
    u = src_combined[:, :, channel]
    f = dst[:, :, channel]

    gi_i = []
    gj_u = []

    for i in src_shifted:
        gi, gj = poisson_editing.im_fwd_gradient(i[:, :, channel])
        gi_i.append(gi)
        gj_u.append(gj)

    gi_f, gj_f = poisson_editing.im_fwd_gradient(f)

    vi, vj = poisson_editing.composite_gradients(gi_i[0], gj_u[0], gi_f, gj_f, mask_shifted_final[0])

    if len(mask_shifted_final) > 1:
        for gi, gj, mask_s in zip(gi_i[1:], gj_u[1:], mask_shifted_final[1:]):       
            vi_aux = vi.copy()
            vj_aux = vj.copy()
            vi, vj = poisson_editing.composite_gradients(gi, gj, vi_aux, vj_aux, mask_s)

    beta_0 = 2.0   # TRY CHANGING
    beta = beta_0 * (1 - m)

    b = poisson_editing.im_bwd_divergence(vi, vj)
    
    x = inpainting.laplace_equation(u, m, b, beta, f, param)

    u_comb[:, :, channel] = x

if padding:
    u_comb = remove_padding(u_comb, org_shape)

u_comb = cv2.normalize(u_comb, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

#zscore_image = (u_comb - np.mean(u_comb)) / np.std(u_comb)
#zscore_normalized = ((zscore_image - np.min(zscore_image)) / (np.max(zscore_image) - np.min(zscore_image)) * 255).astype('uint8')

cv2.imshow('Final result of Poisson blending', u_comb)
cv2.waitKey(0)

cv2.imwrite('./results/'+case + '_result.png', u_comb)

print("FINISHED")
