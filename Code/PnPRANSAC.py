import numpy as np
from numpy.lib.function_base import average
from Code.LinearPnP import linearPnP
from tqdm import tqdm

def computeReprojectionError(X, x, C, R, K):
    C = np.array(C).astype(np.float32).reshape((3,1))
    P_mat = np.matmul(np.matmul(K, R).astype(np.float32), np.hstack((np.eye(3), -1*C)).astype(np.float32)).astype(np.float32)
    u, v = x
    X_x, X_y, X_z = X
    X_homog = np.array([X_x, X_y, X_z, 1]).astype(np.float32)
    term_left = u - (np.dot(P_mat[0], X_homog))/(np.dot(P_mat[2], X_homog))
    term_right = v - (np.dot(P_mat[1], X_homog))/(np.dot(P_mat[2], X_homog))

    error = term_left*term_left + term_right*term_right
    return error

def pnpRANSAC(X, x, K):
    max_iter = 500
    epsilon = 10

    maximized_arguments = linearPnP(X, x, K)
    maximum_num_inliers = 0

    for _ in tqdm(range(max_iter)):

        random_indices = np.random.choice(len(X), 8, replace=False)
        chosen_X = []
        chosen_x = []
        for r_index in random_indices:
            chosen_X.append(X[r_index])
            chosen_x.append(x[r_index])
        chosen_X = np.array(chosen_X).astype(np.float32)
        chosen_x = np.array(chosen_x).astype(np.float32)

        C_vect, R_mat = linearPnP(chosen_X, chosen_x, K)
        #print("C", C_vect)
        #print("R", R_mat)
        num_inliers = 0
        average_error = 0.0
        for i in range(len(X)):
            X_3d = X[i]
            x_proj = x[i]
            error = computeReprojectionError(X_3d, x_proj, C_vect, R_mat, K)
            average_error += error
            if error < epsilon:
                num_inliers += 1

        average_error = average_error / len(X)
        #print("PnP RANSAC avg error", average_error)
        
        if maximum_num_inliers < num_inliers:
            maximum_num_inliers = num_inliers
            maximized_arguments = [C_vect, R_mat]

    return maximized_arguments
