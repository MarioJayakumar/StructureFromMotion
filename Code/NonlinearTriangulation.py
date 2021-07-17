import numpy as np
import scipy.optimize
from tqdm import tqdm

def minimization_function(n_init_3d, K, C1, R1, C2, R2, p1, p2):
    P_mat1 = np.matmul(np.matmul(K, R1).astype(np.float32), np.hstack((np.eye(3), -1*C1)).astype(np.float32)).astype(np.float32)
    P_mat2 = np.matmul(np.matmul(K, R2).astype(np.float32), np.hstack((np.eye(3), -1*C2)).astype(np.float32)).astype(np.float32)
    x, y, z = n_init_3d
    init_3d = np.array([x, y, z, 1]).astype(np.float32)
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]  

    term1_left = x1 - (np.matmul(P_mat1[0, :].T, init_3d))/(np.matmul(P_mat1[2, :].T, init_3d))
    term1_right = y1 - (np.matmul(P_mat1[1, :].T, init_3d))/(np.matmul(P_mat1[2, :].T, init_3d))
    term2_left = x2 - (np.matmul(P_mat2[0, :].T, init_3d))/(np.matmul(P_mat2[2, :].T, init_3d))
    term2_right = y2 - (np.matmul(P_mat2[1, :].T, init_3d))/(np.matmul(P_mat2[2, :].T, init_3d))    
    term1 = term1_left*term1_left + term1_right*term1_right
    term2 = term2_left*term2_left + term2_right*term2_right
    return term1 + term2

def nonLinearTriangulation(K, C1, R1, C2, R2, matches, startingPoints):
    C1 = np.array(C1).astype(np.float32).reshape((3, 1))
    C2 = np.array(C2).astype(np.float32).reshape((3, 1))    
    optimized_points = []
    for i in tqdm(range(len(matches))):
        a, b, c, d = matches[i]
        p1 = (a, b)
        p2 = (c, d)

        init3d = startingPoints[i]

        optimized_args = scipy.optimize.least_squares(fun=minimization_function, x0=init3d, method="dogbox", args=[K, C1, R1, C2, R2, p1, p2])
        optimized_points.append(optimized_args.x)

    return np.array(optimized_points).astype(np.float32)

    