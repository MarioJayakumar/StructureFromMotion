import numpy as np

# fund_mat: fundamental matrix
# K_mat: camera intrinsic parameter matrix
# output: 3x3 Essential matrix
def computeEssentialMatrixFromFundamentalMatrix(fund_mat, K_mat):
    E_mat = np.dot(K_mat.T, np.dot(fund_mat, K_mat))

    E_U, _, E_V = np.linalg.svd(E_mat)

    reconstructed_singular_vals = np.diag([1,1,0]).astype(np.float32)

    E_mat = np.dot(np.dot(E_U, reconstructed_singular_vals), E_V).astype(np.float32)

    return E_mat