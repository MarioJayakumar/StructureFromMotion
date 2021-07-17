import numpy as np

def linearPnP(X, x, K):
    # first we make x homogeneous and normalize
    K_inv = np.linalg.inv(K)
    x_homog = []
    for i in range(len(x)):
        a, b = x[i]
        homog = np.array([a, b, 1]).astype(np.float32)
        homog = np.matmul(K_inv, homog)
        homog = homog/homog[2]
        x_homog.append(homog)

    A_list = []
    for i in range(len(x)):
        X_x, X_y, X_z = X[i]
        u, v, _ = x_homog[i]
        row1 = np.zeros((12,))
        row1[4] = -1*X_x
        row1[5] = -1*X_y
        row1[6] = -1*X_z
        row1[7] = -1

        row1[8] = v*X_x
        row1[9] = v*X_y
        row1[10] = v*X_z
        row1[11] = v      

        row2 = np.zeros((12,))
        row2[0] = X_x
        row2[1] = X_y
        row2[2] = X_z
        row2[3] = 1
        
        row2[8] = -1*u*X_x
        row2[9] = -1*u*X_y
        row2[10] = -1*u*X_z
        row2[11] = -1*u 

        row3 = np.zeros((12,))
        row3[0] = -1*v*X_x
        row3[1] = -1*v*X_y
        row3[2] = -1*v*X_z
        row3[3] = -1*v         

        row3[4] = u*X_x
        row3[5] = u*X_y
        row3[6] = u*X_z
        row3[7] = u   

        A_list.append(row1)
        A_list.append(row2)
        A_list.append(row3)
    
    A_mat = np.array(A_list).astype(np.float32)
    _, _, A_V = np.linalg.svd(A_mat)
    P_vect = A_V[:, -1].reshape((3,4)).astype(np.float32)
    Rot_mat = P_vect[:, 0:3]
    T_vect = P_vect[:, -1]

    # have to correct rot_mat
    R_U, _, R_V = np.linalg.svd(Rot_mat)
    R = np.matmul(R_U, R_V)
    if np.linalg.det(R) < 0:
        R = -1*R
        T_vect = -1*T_vect

    C_vect = -1 * np.matmul(R.T, T_vect)

    return C_vect, R


