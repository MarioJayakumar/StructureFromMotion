import numpy as np 
import cv2


def linearTriangulation2_class(K, C1, R1, C2, R2, matching):
    P_mat1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3), -1*C1))))
    P_mat2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3), -1*C2))))
    x1_set = []
    x2_set = []
    x_set = []
    for match in matching:
        x1, y1, x2, y2 = match
        x1_set.append((x1, y1))
        x2_set.append((x2, y2))
        A_mat = []
        A_mat.append(x1*P_mat1[2] - P_mat1[0])
        A_mat.append(y1*P_mat1[2] - P_mat1[1])
        A_mat.append(x2*P_mat2[2] - P_mat2[0])
        A_mat.append(y2*P_mat2[2] - P_mat2[1])      

        A_mat = np.array(A_mat).astype(np.float32)
        _, _, A_V = np.linalg.svd(A_mat)
        X_vect = A_V[:, -1]
        X_vect = X_vect/X_vect[-1]
        X_vect = X_vect[:3]
        x_set.append(X_vect)

    return x_set  

def linearTriangulation_submitted(K, C1, R1, C2, R2, matching):
    C1 = np.array(C1).astype(np.float32).reshape((3, 1))
    C2 = np.array(C2).astype(np.float32).reshape((3, 1))
    P_mat1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3), -1*C1))))
    P_mat2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3), -1*C2))))
    x_set = np.zeros((len(matching), 3))
    index = 0
    for match in matching:
        x1, y1, x2, y2 = match
        

        lambda_x1 = np.array([[0, -1, y1] ,[1, 0, -x1] ,[-y1, x1, 0]]).astype(np.float32)
        lambda_x2 = np.array([[0, -1, y2] ,[1, 0, -x2] ,[-y2, x2, 0]]).astype(np.float32)

        A_mat = np.vstack((np.dot(lambda_x1, P_mat1), np.dot(lambda_x2, P_mat2)))
        A_mat = np.array(A_mat).astype(np.float32)
        _, _, A_V = np.linalg.svd(A_mat)
        X_vect = A_V[-1]
        X_vect = X_vect/X_vect[-1]
        X_vect = X_vect.reshape((4,1))
        X_vect = X_vect[:3]
        x_set[index,:] = X_vect.reshape((3,))

        index += 1
    return x_set        

"""
Triangulation: 
P1 = K * R1 * [eye(3) -C1]; 
P2 = K * R2 * [eye(3) -C2]; 
N = size(x1, 1); X = zeros(N, 3); 
for i = 1 : N 
    A = [vec2skew([x1(i,:) 1])*P1;... vec2skew([x2(i,:) 1])*P2]; 
    [~, ~, V] = svd(A); 
    X_h = V(:,end)/V(end,end); 
    X(i,:) = X_h(1:3)'; 
end 
function C = vec2skew(v) 
C = [0 -v(3) v(2);... v(3) 0 -v(1);... -v(2) v(1) 0]; 
end end 
"""
def vect2skew(v):
    C_0 = np.array([0, -v[2], v[1]])
    C_1 = np.array([v[2], 0, -v[0]])
    C_2 = np.array([-v[1], v[0], 0])
    C = np.array([C_0, C_1, C_2]).astype(np.float32)
    return C

def linearTriangulation(K, C1, R1, C2, R2, matching):
    C1 = np.array(C1).astype(np.float32).reshape((3, 1))
    C2 = np.array(C2).astype(np.float32).reshape((3, 1))

    P1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3), -C1)))).astype(np.float32)
    P2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3), -C2)))).astype(np.float32)

    x_set = []
    for match in matching:
        x1, y1, x2, y2 = match
        x_vect1 = np.array([x1, y1, 1])
        x_vect2 = np.array([x2, y2, 1])
        A_mat = np.vstack((np.dot(vect2skew(x_vect1), P1), np.dot(vect2skew(x_vect2), P2))).astype(np.float32)
        _, _, V = np.linalg.svd(A_mat)
        V = V[:, -1] / V[-1, -1]
        triangulated = V[:3].reshape((3,))
        x_set.append(triangulated)
    return x_set
    """
FMat calculate: x1 = X1(:,1); y1 = X1(:,2); x2 = X2(:,1); y2 = X2(:,2); A = []; for i = 1:size(X1,1) A(i,:) = [x1(i) * x2(i), x1(i) * y2(i), x1(i) ,y1(i) * x2(i), y1(i) * y2(i), y1(i), x2(i), y2(i), 1]; end [~, ~,V] = svd(A); F = reshape(V(:,9), 3, 3)'; % Rank 2 Fundamental Matrix: [U,S,V] = svd(F,0); F = U * diag([S(1,1), S(2,2), 0]) * V'; % U * (diag(S)) * V'; 
Triangulation: P1 = K * R1 * [eye(3) -C1]; P2 = K * R2 * [eye(3) -C2]; N = size(x1, 1); X = zeros(N, 3); for i = 1 : N A = [vec2skew([x1(i,:) 1])*P1;... vec2skew([x2(i,:) 1])*P2]; [~, ~, V] = svd(A); X_h = V(:,end)/V(end,end); X(i,:) = X_h(1:3)'; end function C = vec2skew(v) C = [0 -v(3) v(2);... v(3) 0 -v(1);... -v(2) v(1) 0]; end end 
    """