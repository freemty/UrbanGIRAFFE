import numpy as np
import math
from numpy import sin , cos

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    R = R[:3,:3]
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-3

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotate_mat2Euler(rotate_mat):


    # if len(rotate_mat.shape) == 2:
    #     rotate_mat = rotate_mat[None]
    # batch_size = rotate_mat.shape[0]
    # a = [isRotationMatrix(rotate_mat[i])for i in range(batch_size)]
    assert(isRotationMatrix(rotate_mat))
    
    sy = np.sqrt(rotate_mat[0,0] * rotate_mat[0,0] +  rotate_mat[1,0] * rotate_mat[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(rotate_mat[2,1] , rotate_mat[2,2])
        y = math.atan2(-rotate_mat[2,0], sy)
        z = math.atan2(rotate_mat[1,0], rotate_mat[0,0])
    else :
        x = math.atan2(-rotate_mat[1,2], rotate_mat[1,1])
        y = math.atan2(-rotate_mat[2,0], sy)
        z = 0

    return np.array([x, y, z])


def parse_R(R, return_eular_angle = False):
    """ Get rotate and scale from R matrix

    Theory
    a. To Rotate Mat: Rotate^T = Rotate^-1
    b. To Scale  MAt: Scale^T = Scale
    c. (Rotate * Scale)^T= (Scale^T) * (Rotate^T) = Scale * (Rotate.-1)
    d. A * A^-1 = E 
    e. Thus: R^T * R = (Roate * Scale)^T * (Rotate * Scale) = Scale * (Rotate^-1) * Rotate * Scale * = Scale * E * Scale = Scale * Scale
    """
    flag = 0
    if len(R.shape) == 2:
        R = R[None]
        flag = 1
    R = R[:,:3,:3]
    batch_size = R.shape[0]
    scale_mat = R.transpose(0,2,1) @ R
    sx = np.sqrt(scale_mat[:,0,0])
    sy = np.sqrt(scale_mat[:,1,1])
    sz = np.sqrt(scale_mat[:,2,2])
    scale_mat = np.zeros_like(scale_mat)
    scale_mat[:,0,0],scale_mat[:,1,1],scale_mat[:,2,2] = sx,sy,sz
    scale = np.concatenate((sx[:,None], sy[:,None], sz[:,None]),axis=-1)
    
    #rotate
    rotate_mat = np.matmul(R, np.linalg.inv(scale_mat))
    assert([isRotationMatrix(rotate_mat[i])for i in range(batch_size)])
    eular_angle = np.stack([rotate_mat2Euler(rotate_mat[i])  for i in range(batch_size)])
    if flag == 1:
        return scale[0], rotate_mat[0] , eular_angle[0]
    else:
        return scale, rotate_mat , eular_angle

def create_R(rotate = (0,0,0), scale = (1,1,1)):
    """ Build R matrix from rotate(eular angle) and scale(xyz)
    Args:
        rotate: eular angle (alpha, beta, gamma)
        scale: (s_x, s_y, s_z)
    Return:
        R
    """
    alpha, beta, gamma = rotate[0], rotate[1], rotate[2]

    rx_mat = np.array([
        [1,0,0],
        [0,cos(alpha),-sin(alpha)],
        [0,sin(alpha),cos(alpha)],   
    ])
    ry_mat = np.array([
        [cos(beta),0,sin(beta)],
        [0,1,0],
        [-sin(beta),0,cos(beta)],
    ])

    rz_mat = np.array([
        [cos(gamma),-sin(gamma),0],
        [sin(gamma),cos(gamma),0],
        [0,0,1],
    ])

    scale_mat = np.array([
        [scale[0],0,0],
        [0,scale[1],0],
        [0,0,scale[2]],
    ])

    R = np.matmul(rz_mat, ry_mat).dot(rx_mat).dot(scale_mat)

    return R


def tr2RT(tr):
    assert tr.shape == (4,4)
    R = tr[0:3,0:3]
    T = tr[0:3, 3]
    return R, T

def RT2tr(R, T):
    assert R.shape == (3,3)
    tr = np.zeros((4,4))
    tr[3, 3] = 1
    tr[0:3, 0:3] = R
    tr[0:3, 3] = T
    return tr
