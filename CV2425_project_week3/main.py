import cv2
import numpy as np
from tqdm import tqdm
from utils import *

folderInput = 'images/'
figure_name = 'circles.png'
figure_name_final = folderInput + figure_name
img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)

# Visualize the grayscale image
# cv2.imshow('Image', img); cv2.waitKey(0)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
# cv2.imshow('Normalized image',img)
# cv2.waitKey(0)

# Height and width
ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
# if len(img.shape) > 2:
#     nc = img.shape[2] # number of channels
#     img = np.mean(img, axis=2)


# Try out different parameters
mu = 0.2
nu = 0.0
lambda1 = 1.0
lambda2 = 1.0
tol = 10e-3
dt = 0.5
iterMax = 30000
pattern = 'circle' #checkboard or circle


X, Y = np.meshgrid(np.arange(0, nj), np.arange(0, ni), indexing='xy')

# Initial phi
# This initialization allows a faster convergence for phantom2
if pattern == 'circle':
    phi = (-np.sqrt((X - np.round(ni / 2)) ** 2 + (Y - np.round(nj / 2)) ** 2) + 50)

elif pattern == 'checkboard':
    phi = -np.sin((np.pi / 5) * X)*np.sin((np.pi / 5) * Y) #Checkboard pattern

# Normalization of the initial phi to the range [-1, 1]
min_val = np.min(phi)
max_val = np.max(phi)
# phi = phi - min_val
# phi = 2 * phi / max_val
# phi = phi - 1

phi = 2 * (phi - min_val) / (max_val - min_val) - 1


# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)

# Gradient descent for the level set evolution
for i, it in enumerate(tqdm(range(iterMax))):

    previous_phi = phi.copy()
    
    # Boundary conditions
    phi[0, :] = phi[1, :]  
    phi[-1, :] = phi[-2, :] 

    phi[:, 0] = phi[:, 1] 
    phi[:, -1] = phi[:, -2]  

    # Compute Heaviside and Dirac delta functions
    H_phi = heaviside(phi)
    delta_phi = delta(phi)

    # Update region averages c1 and c2    
    # Compute the data fitting term
    if len(img.shape) >2: #For RGB images
        inside_term = 0 
        outside_term = 0 

        for channel in range(img.shape[2]):
            c1 = np.sum(img[:,:,channel] * H_phi) / np.sum(H_phi)
            c2 = np.sum(img[:,:,channel] * (1 - H_phi)) / np.sum((1 - H_phi))
        
            inside_term += lambda1 * (img[:,:,channel] - c1) ** 2
            outside_term += lambda2 * (img[:,:,channel] - c2) ** 2
        
        inside_term = inside_term/img.shape[2]
        outside_term = outside_term/img.shape[2]

    else: #For greyscale images
        c1 = np.sum(img * H_phi) / np.sum(H_phi)
        c2 = np.sum(img * (1 - H_phi)) / np.sum((1 - H_phi))
        inside_term = lambda1 * (img - c1) ** 2
        outside_term = lambda2 * (img - c2) ** 2

    fwd = im_fwd_gradient(phi)
    bwd = im_bwd_gradient(phi)

    bwd_fwd_x, _ = im_bwd_gradient(fwd[0])
    _, bwd_fwd_y = im_bwd_gradient(fwd[1])

    A = fwd[0] / np.sqrt(((10e-8)**2 + (fwd[0])**2 + (((fwd[1]+bwd[1])/2))**2))
    B = fwd[1] / np.sqrt(((10e-8)**2 + (fwd[1])**2 + (((fwd[0]+bwd[0])/2))**2))

    bwd_fwd_x, _ = im_bwd_gradient(A)
    _, bwd_fwd_y = im_bwd_gradient(B)
    
    # Update phi using the gradient descent equation
    dphi_dt = delta_phi * (mu*(bwd_fwd_x+bwd_fwd_y) - nu - inside_term + outside_term)
    phi = phi + dt * dphi_dt
    
    # Check for convergence
    check = np.linalg.norm(phi-previous_phi) #L2 norm
    
    if check <= tol:
        print(f'Converged after {it} iterations')
        break
    
    #Show progress
    if i%10 == 0:

        seg = np.where(phi <= 0, 0, 1).astype('float')
        seg = (seg * 255).astype(np.uint8)  # Convert to uint8 for display
        cv2.imshow('Iterations', seg)
        cv2.waitKey(1)

        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_normalized.astype(np.uint8)

        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        
        contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)
        cv2.imshow('Contours', img_color)
        cv2.waitKey(1)

# Segment the image based on the final phi
seg = np.where(phi <= 0, 0, 1)

# # CODE TO COMPLETE
# Show final segmented image 
cv2.imshow('Final segmentation', seg.astype('float'))
cv2.waitKey(0)

if len(img.shape) == 2:
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
else:
    img_color = img.copy()

contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)
cv2.imshow('Final contours', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

