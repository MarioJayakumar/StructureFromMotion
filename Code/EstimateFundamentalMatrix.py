import numpy as np
import cv2
from numpy.lib.utils import source
import matplotlib.pyplot as plt
from Code.Utils import plot_images

# source points is list of (x,y)
# dest points is list of (x,y)
# returns 3x3 matrix
def computeFundamentalMatrix(source_points, dest_points, src_image=None, dst_image=None):
    num_correspondence = len(source_points)
    A = []

    source_points = np.array(source_points).astype(np.float32)
    dest_points = np.array(dest_points).astype(np.float32)

    source_mean_x = np.mean(source_points[:,0])
    source_mean_y = np.mean(source_points[:,1])
    dest_mean_x = np.mean(dest_points[:,0])
    dest_mean_y = np.mean(dest_points[:,1])  

    denom1 = 0.0
    denom2 = 0.0


    for i in range(len(source_points)):
        x1, y1 = source_points[i]
        x2, y2 = dest_points[i]

        x1_new = x1-source_mean_x
        y1_new = y1-source_mean_y

        x2_new = x2-dest_mean_x
        y2_new = y2-dest_mean_y

        denom1 += x1_new**2 + y1_new**2
        denom2 += x2_new**2 + y2_new**2

    denom1 = denom1/len(source_points)
    denom2 = denom2/len(source_points)

    scale1 = np.sqrt(2/denom1)
    scale2 = np.sqrt(2/denom2)

    T_mat1_left = np.diag([scale1, scale1, 1]).astype(np.float32)
    T_mat1_right = np.array([[1, 0, -1*source_mean_x] ,[0, 1, -1*source_mean_y] ,[0, 0, 1]]).astype(np.float32)
    T_mat2_left = np.diag([scale2, scale2, 1]).astype(np.float32)
    T_mat2_right = np.array([[1, 0, -1*dest_mean_x] ,[0, 1, -1*dest_mean_y] ,[0, 0, 1]]).astype(np.float32)    

    T_mat1 = np.matmul(T_mat1_left, T_mat1_right)
    T_mat2 = np.matmul(T_mat2_left, T_mat2_right)

    stand_source_points = []
    stand_dest_points = []
    for i in range(len(source_points)):
        x1, y1 = source_points[i]
        x2, y2 = dest_points[i]

        ve1 = np.array([x1, y1, 1]).astype(np.float32)
        ve2 = np.array([x2, y2, 1]).astype(np.float32)

        standard_ve1 = np.matmul(T_mat1, ve1)
        standard_ve2 = np.matmul(T_mat2, ve2)

        sx1 = standard_ve1[0]/standard_ve1[2]
        sy1 = standard_ve1[1]/standard_ve1[2]        
        sx2 = standard_ve2[0]/standard_ve2[2]
        sy2 = standard_ve2[1]/standard_ve2[2]   

        stand_source_points.append((sx1, sy1))
        stand_dest_points.append((sx2, sy2))


    for index in range(num_correspondence):
        x, y = stand_source_points[index]
        x_prime, y_prime = stand_dest_points[index]
        row = np.array([x*x_prime, x*y_prime, x, y*x_prime, y*y_prime, y, x_prime, y_prime, 1])
        A.append(row)
    A = np.array(A).astype(np.float32)
    A_U, A_S, A_V = np.linalg.svd(A)
    F_vect = A_V[:, -1]
    F_mat = F_vect.reshape((3,3)).T.astype(np.float32)
    F_mat = np.matmul(T_mat2.T, np.matmul(F_mat, T_mat1))

    if src_image is not None:
        lines1 = cv2.computeCorrespondEpilines(dest_points.reshape(-1, 1, 2), 2, F_mat)
        lines1 = lines1.reshape(-1, 3)
        new_src, _ = drawLines(src_image, dst_image, lines1, source_points, dest_points)

        lines2 = cv2.computeCorrespondEpilines(source_points.reshape(-1, 1, 2), 1, F_mat)
        lines2 = lines2.reshape(-1, 3)
        new_dst, _ = drawLines(dst_image, src_image, lines2, dest_points, source_points)
        r = np.random.randint(0, 100)
        plot_images([new_src, new_dst], [])
        #cv2.waitKey(0)

    F_U, F_S, F_V = np.linalg.svd(F_mat)
    F_S = np.diag(F_S)
    F_S[2,2] = 0
    F_mat = np.matmul(F_U, np.matmul(F_S, F_V))

    F_mat = F_mat / F_mat[2,2]

    #print(np.linalg.matrix_rank(F_mat))
    return F_mat

"""
FMat calculate: 
x1 = X1(:,1); 
y1 = X1(:,2); 
x2 = X2(:,1); 
y2 = X2(:,2); 
A = []; 
for i = 1:size(X1,1) 
    A(i,:) = [x1(i) * x2(i), x1(i) * y2(i), x1(i) ,y1(i) * x2(i), y1(i) * y2(i), y1(i), x2(i), y2(i), 1]; 
end 
[~, ~,V] = svd(A); 
F = reshape(V(:,9), 3, 3)'; 
% Rank 2 Fundamental Matrix: 
[U,S,V] = svd(F,0); 
F = U * diag([S(1,1), S(2,2), 0]) * V'; 
% U * (diag(S)) * V';
"""
def computeFundamentalMatrix_ref(source_points, dest_points, src_image=None, dst_image = None):
    source_points = np.array(source_points)
    dest_points = np.array(dest_points)
    x1 = source_points[:, 0]
    y1 = source_points[:, 1]
    x2 = dest_points[:, 0]
    y2 = dest_points[:, 1]
    A = []
    for i in range(len(x1)):
        A.append(np.array([x1[i]*x2[i], x1[i]*y2[i], x1[i], y1[i]*x2[i], y1[i]*y2[i], y1[i], x2[i], y2[i], 1]))
    A = np.array(A).astype(np.float32)
    _, _, V = np.linalg.svd(A)
    #F = V[:, -1].reshape((3,3))

    F = computeFundamentalMatrix(source_points, dest_points)

    if src_image is not None:
        lines1 = cv2.computeCorrespondEpilines(dest_points.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        new_src, _ = drawLines(src_image, dst_image, lines1, source_points, dest_points)

        lines2 = cv2.computeCorrespondEpilines(source_points.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        _, new_dst = drawLines(dst_image, src_image, lines2, dest_points, source_points)
        plt.imshow(new_src)
        plt.imshow(new_dst)
        cv2.waitKey(0)

    F_U, F_S, F_V = np.linalg.svd(F)
    F = np.dot(F_U, np.dot(np.diag([F_S[0], F_S[1], 0]), F_V))
    return F

def computeFundamentalMatrix_old(source_points, dest_points):
    pts1 = []
    pts2 = []
    for i in range(len(source_points)):
        pts1.append(source_points[i])
        pts2.append(dest_points[i])

    pts1 = np.array(pts1).astype(np.float32)
    pts2 = np.array(pts2).astype(np.float32)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F

def drawLines(img1, img2, lines, pts1, pts2):
    img1 = img1.copy()
    img2 = img2.copy()
    r, c, _ = img1.shape
    np.random.seed(10)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple([int(pt1[1]), int(pt1[0])]), 5, color, -1)
        #img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2