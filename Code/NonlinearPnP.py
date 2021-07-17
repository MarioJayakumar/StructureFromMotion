import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize

def nonlinearPnP(X, x, C, R, K):
    # generate quaternion from R
    quaternion = Rotation.from_matrix(R)
    quaternion = quaternion.as_quat()

    PoseInit = [C[0], C[1], C[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

    optimized_args = scipy.optimize.least_squares(fun=minimization_function, x0=PoseInit, method='dogbox', args=[X, x, K])
    optim_pose = optimized_args.x
    C_optim = np.zeros((3,))
    C_optim[0] = optim_pose[0]
    C_optim[1] = optim_pose[1]
    C_optim[2] = optim_pose[2]    
    C_optim = C_optim.astype(np.float32)

    R_optim = Rotation.from_quat((optim_pose[3], optim_pose[4], optim_pose[5], optim_pose[6]))
    R_optim = np.array(R_optim.as_matrix()).astype(np.float32)

    return C_optim, R_optim

def minimization_function(Pose, X, x, K):
    C = np.zeros((3,))
    C[0] = Pose[0]
    C[1] = Pose[1]
    C[2] = Pose[2]
    C  = C.astype(np.float32).reshape((3,1))

    R_mat = Rotation.from_quat((Pose[3], Pose[4], Pose[5], Pose[6]))
    R_mat = np.array(R_mat.as_matrix()).astype(np.float32)

    P_mat = np.matmul(np.matmul(K, R_mat).astype(np.float32), np.hstack((np.eye(3), -1*C)).astype(np.float32)).astype(np.float32)

    error = 0.0
    for i in range(len(X)):
        X_x, X_y, X_z = X[i]
        X_homog = np.array([X_x, X_y, X_z, 1]).astype(np.float32)
        u, v = x[i]
        term_left = u - (np.dot(P_mat[0], X_homog))/(np.dot(P_mat[2], X_homog))
        term_right = v - (np.dot(P_mat[1], X_homog))/(np.dot(P_mat[2], X_homog))

        error += term_left**2 + term_right**2

    return error