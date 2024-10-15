from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class Parameters:
    hi: float
    hj: float

def laplace_equation(f, mask, grads, beta, dst, param):

    ni = f.shape[0] #Number of rows
    nj = f.shape[1] #Number of columns

    f_ext = f

    mask_ext = mask

    # Store memory for the A matrix and the b vector
    nPixels = ni*nj # Number of pixels

    b = np.zeros(nPixels, dtype=float)

    beta = beta.reshape(-1, order='F')

    # Vector counter
    idx=0
    idx_Ai=[]
    idx_Aj=[]
    a_ij=[]

    # Loop over the pixels to identify the boundary pixels
    for j in range(0, nj ):
        for i in range(0, ni):
            if mask_ext[i, j] == 1:  # This pixel is part of the mask (inside the mask)

                # Check for neighbors that are outside the mask (i.e., boundary)
                # This pixel is on the boundary if any neighbor is outside the mask
                if mask_ext[i+1, j] == 0:  # Neighbor to the north is outside the mask
                    p_out = (j) * (ni ) + (i)  # Outside pixel (north)
                    p_in = (j) * (ni ) + i +1 # Inside pixel

                    # Apply 1 to the outside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out)
                    a_ij.append(1)

                    # Apply -1 to the inside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_in)
                    a_ij.append(-1)
                    
                    b[p_out] = 0  # Dirichlet boundary condition (0 in this case)

                if mask_ext[i-1, j] == 0:  # Neighbor to the south is outside the mask
                    p_out = (j) * (ni ) + (i)  # Outside pixel (south)
                    p_in = (j) * (ni ) + i-1  # Inside pixel

                    # Apply 1 to the outside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out)
                    a_ij.append(1)

                    # Apply -1 to the inside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_in)
                    a_ij.append(-1)

                    b[p_out] = 0  # Dirichlet boundary condition (0 in this case)

                if mask_ext[i, j+1] == 0:  # Neighbor to the east is outside the mask
                    p_out = (j) * (ni ) + i  # Outside pixel (east)
                    p_in = (j+1) * (ni ) + i  # Inside pixel

                    # Apply 1 to the outside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out)
                    a_ij.append(1)

                    # Apply -1 to the inside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out - (ni ))
                    a_ij.append(-1)

                    b[p_out] = 0  # Dirichlet boundary condition (0 in this case)

                if mask_ext[i, j-1] == 0:  # Neighbor to the west is outside the mask
                    p_out = (j-1) * (ni) + i  # Outside pixel (west)
                    p_in = j * (ni) + i  # Inside pixel

                    # Apply 1 to the outside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out)
                    a_ij.append(1)

                    # Apply -1 to the inside pixel
                    idx_Ai.append(p_out)
                    idx_Aj.append(p_out + (ni ))
                    a_ij.append(-1)

                    b[p_out] = 0  # Dirichlet boundary condition (0 in this case)

    # Looping over the pixels
    for j in range(0, nj):
        for i in range(0, ni):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = j * (ni) + i

            if mask_ext[i, j] == 1: # we have to in-paint this pixel

                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p)
                a_ij.insert(idx,  4)  # Laplace operator central term
                idx += 1

                # Link to north neighbor
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p + 1)
                a_ij.insert(idx,  -1)
                idx += 1

                # Link to south neighbor
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p - 1)
                a_ij.insert(idx,  -1)
                idx += 1

                # Link to west neighbor
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p + (ni))
                a_ij.insert(idx,  -1)
                idx += 1

                # Link to east neighbor
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p - (ni))
                a_ij.insert(idx,  -1)
                idx += 1

                # Right-hand side of the equation
                b[p] =  -grads[i,j]  # In-painting

            else: # we do not have to in-paint this pixel

                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p)
                a_ij.insert(idx, beta[p])
                idx += 1

                b[p] = beta[p]*f_ext[i, j]  # Keep the original value

    idx_Ai_c = idx_Ai
    idx_Aj_c = idx_Aj

    # COMPLETE THE CODE (fill out the interrogation marks ???)
    A = sparse(idx_Ai_c, idx_Aj_c, a_ij, nPixels, nPixels) #It is nPixels because A matrix should have shape (ni+2)*(nj+2)
    x = spsolve(A, b)

    u_ext = np.reshape(x,(ni, nj), order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    u = np.full((ni, nj), u_ext[0:u_ext_i, 0:u_ext_j], order='F')
    return u

def sparse(i, j, v, m, n):
    """
    Create and compress a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))