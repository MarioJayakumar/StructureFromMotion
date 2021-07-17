import numpy as np

def passesChiralityCheck(R_mat, C_vect, point):
    C_vect = np.array(C_vect).reshape((3,1))
    point = np.array(point).reshape((3,1))
    right = C_vect - point
    r3 = R_mat[2].reshape((3,1))
    chiral = np.dot(r3.T, right)
    if chiral > 0:
        return True
    return False

def disambiguateCameraPose(C_set, R_set, X_set):
    maximal_points = 0
    best_index = 0

    for i in range(4):
        C_vect = C_set[i]
        R_mat = R_set[i]
        X_points = X_set[i]

        passed = 0

        for point in X_points:
            if passesChiralityCheck(R_mat, C_vect, point):
                passed += 1
        
        if passed > maximal_points:
            maximal_points = passed
            best_index = i

    return C_set[best_index], R_set[best_index], X_set[best_index]

