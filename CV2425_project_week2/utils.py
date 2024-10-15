import numpy as np

# Function to shift the source image to the destination position
def shift_image(src, mask, translation):
    shifted_src = np.zeros_like(src)
    shifted_mask = np.zeros_like(mask)
    dy, dx = translation

    ni, nj, nChannels = src.shape

    # Shift the mask and the source
    shifted_src[max(0, dy):ni, max(0, dx):nj] = src[max(0, -dy):ni-dy, max(0, -dx):nj-dx]
    shifted_mask[max(0, dy):ni, max(0, dx):nj] = mask[max(0, -dy):ni-dy, max(0, -dx):nj-dx]

    return shifted_src, shifted_mask

def combine_sources_with_masks(src, src2, dst, mask_shifted_eyes, mask_shifted_mouth):

    # Combine masks by taking the maximum (union of both masks)
    combined_mask = np.maximum(mask_shifted_eyes, mask_shifted_mouth)

    # Initialize combined image with destination as the background
    combined_src = np.copy(dst)

    # For each channel, apply the source image only where the mask is active
    for i in range(3):  # Loop over RGB channels
        combined_src[:, :, i] = np.where(mask_shifted_eyes[:, :, i] > 0, src[:, :, i], combined_src[:, :, i])
        combined_src[:, :, i] = np.where(mask_shifted_mouth[:, :, i] > 0, src2[:, :, i], combined_src[:, :, i])

    return combined_src, combined_mask