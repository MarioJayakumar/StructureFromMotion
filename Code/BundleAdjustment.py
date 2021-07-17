import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize

def bundleAdjustment(C_set, R_set, X_set, correspondence, vis_mat, K):
    init_pose = np.zeros((7*len(C_set) + 3*len(X_set), 1))
    for i in range(len(C_set)):
        quaternion = Rotation.from_matrix(R_set[i])
        quaternion = quaternion.as_quat()

        init_pose[0 + 7*i] = C_set[i][0]
        init_pose[1 + 7*i] = C_set[i][1]
        init_pose[2 + 7*i] = C_set[i][2]

        init_pose[3 + 7*i] = quaternion[0]
        init_pose[4 + 7*i] = quaternion[1]
        init_pose[5 + 7*i] = quaternion[2]
        init_pose[6 + 7*i] = quaternion[3]

    temp_X_set = np.array(X_set).astype(np.float32).reshape((-1, 1))
    init_pose[(7*len(C_set)):] = temp_X_set
    init_pose = np.array(init_pose).astype(np.float32).flatten()
    optimized_args = scipy.optimize.least_squares(fun=minimization_function, x0=init_pose, method="dogbox", args=[correspondence, K, vis_mat])
    optim_pose = optimized_args.x

    new_C_set = []
    new_R_set = []
    new_X_set = []

    pose_info = optim_pose[:(7*len(C_set))]
    for i in range(len(C_set)):
        C_optim = np.zeros((3,))
        C_optim[0] = pose_info[0 + 7*i]
        C_optim[1] = pose_info[1+ 7*i]
        C_optim[2] = pose_info[2 + 7*i]   
        C_optim = C_optim.astype(np.float32)

        R_optim = Rotation.from_quat((pose_info[3 + 7*i], pose_info[4 + 7*i], pose_info[5 + 7*i], pose_info[6 + 7*i]))
        R_optim = np.array(R_optim.as_matrix()).astype(np.float32)

        new_C_set.append(C_optim)
        new_R_set.append(R_optim)

    x_set_info = optim_pose[(7*len(C_set)):]
    for i in range(len(X_set)):
        sub_x = x_set_info[i*3]
        sub_y = x_set_info[1 + i*3]
        sub_z = x_set_info[2 + i*3]
        new_X_set.append((sub_x, sub_y, sub_z))
    new_X_set = np.array(new_X_set).astype(np.float32)

    return new_C_set, new_R_set, new_X_set

def minimization_function(Pose, correspondence, K, vis_mat):
    C_set = []
    R_set = []
    set_size = len(correspondence)
    for i in range(set_size):
        C = np.zeros((3,))
        C[0] = Pose[0 + 7*i]
        C[1] = Pose[1 + 7*i]
        C[2] = Pose[2 + 7*i]
        C  = C.astype(np.float32).reshape((3,1))

        R_mat = Rotation.from_quat((Pose[3 + 7*i], Pose[4 + 7*i], Pose[5 + 7*i], Pose[6 + 7*i]))
        R_mat = np.array(R_mat.as_matrix()).astype(np.float32)
        C_set.append(C)
        R_set.append(R_mat)

    X = []
    sub_pose = Pose[(7*len(correspondence)):]
    for i in range(int(len(sub_pose)/3)):
        sub_x = sub_pose[3*i]
        sub_y = sub_pose[1 + 3*i]
        sub_z = sub_pose[2 + 3*i]
        X.append((sub_x, sub_y, sub_z))

    error = 0.0
    for j in range(set_size):
        R_mat = R_set[j]
        C = C_set[j]
        P_mat = np.matmul(np.matmul(K, R_mat).astype(np.float32), np.hstack((np.eye(3), -1*C)).astype(np.float32)).astype(np.float32)

        error = 0.0
        for i in range(len(correspondence[j])):
            mapping_point = correspondence[j][i]
            source_tuple = list(mapping_point.keys())[0]
            dest_index = list(mapping_point.values())[0]
            if vis_mat[j][dest_index] == 1:
                X_x, X_y, X_z = X[dest_index]
                X_homog = np.array([X_x, X_y, X_z, 1]).astype(np.float32)
                u, v = source_tuple
                term_left = u - (np.dot(P_mat[0], X_homog))/(np.dot(P_mat[2], X_homog))
                term_right = v - (np.dot(P_mat[1], X_homog))/(np.dot(P_mat[2], X_homog))

                error += term_left**2 + term_right**2

    return error