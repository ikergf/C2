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

def combine_sources_with_masks(src, dst, mask):

    if len(mask)>1:
        # Combine masks by taking the maximum (union of both masks)
        combined_mask = np.maximum(mask[0], mask[1])
        print('a')
    else:
        combined_mask = mask[0]

    # Initialize combined image with destination as the background
    combined_src = np.copy(dst)
    print(src[0].shape)

    # For each channel, apply the source image only where the mask is active
    for i in range(3):  # Loop over RGB channels
        if len(src)>1:
            combined_src[:, :, i] = np.where(mask[0][:, :, i] > 0, src[0][:, :, i], combined_src[:, :, i])
            combined_src[:, :, i] = np.where(mask[1][:, :, i] > 0, src[1][:, :, i], combined_src[:, :, i])
        else:
            combined_src[:, :, i] = np.where(mask[0][:, :, i] > 0, src[0][:, :, i], combined_src[:, :, i])

    return combined_src, combined_mask