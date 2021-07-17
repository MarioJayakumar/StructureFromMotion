import numpy as np

# E_Mat: 3x3 Essential Matrix
# Output: C_list: list where row i is 3d vect C_i
# Output: R_list: list where row i is 3x3 matrix R_i
def extractCameraPose(E_mat):

    E_U, _, E_V = np.linalg.svd(E_mat)
    W_mat = np.array([[0, -1, 0] ,[1, 0, 0] ,[0, 0, 1]]).astype(np.float32)

    C_list = []
    R_list = []

    C_list.append(E_U[:, 2])
    C_list.append(-1*E_U[:, 2])
    C_list.append(E_U[:, 2])
    C_list.append(-1*E_U[:, 2])
    R_list.append(np.dot(E_U, np.dot(W_mat, E_V)))
    R_list.append(np.dot(np.dot(E_U, W_mat), E_V))
    R_list.append(np.dot(np.dot(E_U, W_mat.T), E_V))
    R_list.append(np.dot(np.dot(E_U, W_mat.T), E_V))

    for i in range(4):
        if np.linalg.det(R_list[i]) < 0:
            C_list[i] = -1*C_list[i]
            R_list[i] = -1*R_list[i]

    return C_list, R_list