import cv2
import numpy as np
from dataclasses import dataclass
import inpainting

@dataclass
class Parameters:
    hi: float
    hj: float

# Folder with the images
image_folder = 'images/'
for i in range(4):
    image_name = f'image{i+1}'
    # There are several black and white images to test inside the images folder:
    #  image1_to_restore.jpg
    #  image2_to_restore.jpg
    #  image3_to_restore.jpg
    #  image4_to_restore.jpg
    #  image5_to_restore.jpg

    # Read an image to be restored
    full_image_path = image_folder + image_name + '_to_restore.jpg'
    im = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)

    # Print image dimensions
    print('Image Dimensions : ', im.shape)
    print('Image Height     : ', im.shape[0])
    print('Image Width      : ', im.shape[1])

    # Show image
    #cv2.imshow('Original image', im)
    #cv2.waitKey(0)

    # Normalize values into [0,1]
    min_val = np.min(im)
    max_val = np.max(im)
    im = (im.astype('float') - min_val)
    im = im / max_val

    # Show normalized image
    #cv2.imshow('Normalized image', im)
    #cv2.waitKey(0)

    # Load the mask image
    full_mask_path = image_folder + image_name + '_mask.jpg'
    mask_img = cv2.imread(full_mask_path, cv2.IMREAD_UNCHANGED)
    # From the mask image we define a binary mask that "erases" the darker pixels from the original image
    mask = (mask_img > 128).astype('float')
    # mask[i,j] == 1 means we have lost information in that pixel
    # mask[i,j] == 0 means we have information in that pixel
    # We want to in-paint those areas in which mask == 1

    # Mask dimensions
    dims = mask.shape
    ni = mask.shape[0]
    nj = mask.shape[1]
    print('Mask Dimension : ', dims)
    print('Mask Height    : ', ni)
    print('Mask Width     : ', nj)

    # Show the mask image and binarized mask
    #cv2.imshow('Mask image', mask_img)
    #cv2.waitKey(0)
    #cv2.imshow('Binarized mask', mask)
    #cv2.waitKey(0)

    # Parameters
    param = Parameters(0, 0)
    param.hi = 1 / (ni-1)
    param.hj = 1 / (nj-1)

    # Perform the in-painting
    u = inpainting.laplace_equation(im, mask, param)

    # Show the final image
    #cv2.imshow('In-painted image', u)
    #cv2.waitKey(0)
    cv2.imwrite(image_name + '_inpainted.png', cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    del im, u, mask_img

### Let us now try with a colored image (image6) ###
for i in range(5,7):
    
    image_name = f'image{i}'
    if i == 5:
        ext = '.jpg'
    elif i == 6:
        ext = '.tif'
    full_image_path = image_folder + image_name + '_to_restore' + ext
    im = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)

    # Normalize values into [0,1]
    min_val = np.min(im)
    max_val = np.max(im)
    im = (im.astype('float') - min_val)
    im = im / max_val

    # Show normalized image
    #cv2.imshow('Normalized Image', im)
    #cv2.waitKey(0)

    # Number of pixels for each dimension, and number of channels
    # height, width, number of channels in image
    ni = im.shape[0]
    nj = im.shape[1]
    nc = im.shape[2]

    # Load and show the (binarized) mask
    full_mask_path = image_folder + image_name + '_mask' + ext
    mask_img = cv2.imread(full_mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask_img > 128).astype('float')
    #cv2.imshow('Binarized mask', mask)
    #cv2.waitKey(0)

    # Parameters
    param = Parameters(0, 0)
    param.hi = 1 / (ni-1)
    param.hj = 1 / (nj-1)

    # Perform the in-painting for each channel separately
    u = np.zeros(im.shape, dtype=float)
    u[:, :, 0] = inpainting.laplace_equation(im[:, :, 0], mask[:, :, 0], param)
    u[:, :, 1] = inpainting.laplace_equation(im[:, :, 1], mask[:, :, 1], param)
    u[:, :, 2] = inpainting.laplace_equation(im[:, :, 2], mask[:, :, 2], param)

    # Show the final image
    #cv2.imshow('In-painted image', u)
    #cv2.waitKey(0)

    cv2.imwrite(image_name + '_inpainted.png', cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    del im, u, mask_img

### Let us now try with a colored image (image7) without a mask image ###

# Write your code to remove the red text overlayed on top of image7_to_restore.png
# Hint: the undesired overlay is plain red, so it should be easy to extract the (binarized) mask from the image file

### Let us now try with a colored image (image6) ###
for i in range(10,11):
    
    image_name = f'image{i}'

    full_image_path = image_folder + image_name + '_to_restore.png'
    im = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)

    # Create and show the (binarized) mask
    red_mask = np.all(im[:,:,:3] == np.array([0, 0, 255]), axis=-1)
    green_mask = np.all(im[:,:,:3] == np.array([0, 255, 0]), axis=-1)
    blue_mask = np.all(im[:,:,:3] == np.array([255, 0, 0]), axis=-1)

    combined_mask = (red_mask | green_mask | blue_mask).astype('float')

    cv2.imshow('Binarized mask', combined_mask)
    cv2.waitKey(0)
    
    # Normalize values into [0,1]
    min_val = np.min(im)
    max_val = np.max(im)
    im = (im.astype('float') - min_val)
    im = im / max_val

    # Show normalized image
    cv2.imshow('Normalized Image', im)
    cv2.waitKey(0)

    # Number of pixels for each dimension, and number of channels
    # height, width, number of channels in image
    ni = im.shape[0]
    nj = im.shape[1]
    nc = im.shape[2]

    # Parameters
    param = Parameters(0, 0)
    param.hi = 1 / (ni-1)
    param.hj = 1 / (nj-1)

    # Perform the in-painting for each channel separately
    u = np.zeros(im.shape, dtype=float)
    u[:, :, 0] = inpainting.laplace_equation(im[:, :, 0], combined_mask, param)
    u[:, :, 1] = inpainting.laplace_equation(im[:, :, 1], combined_mask, param)
    u[:, :, 2] = inpainting.laplace_equation(im[:, :, 2], combined_mask, param)
    u[:, :, 3] = inpainting.laplace_equation(im[:, :, 3], combined_mask, param)

    # Show the final image
    cv2.imshow('In-painted image', u)
    cv2.waitKey(0)

    cv2.imwrite(image_name + '_inpainted.png', cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    del im, u, combined_mask