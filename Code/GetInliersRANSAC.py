import numpy as np
from tqdm import tqdm
from .EstimateFundamentalMatrix import computeFundamentalMatrix

# remove one-to-many matches
def duplicate_refinement(matches):
    duplicate_indices = []

    for i in range(len(matches)-1):
        if i not in duplicate_indices:
            for j in range(i+1, len(matches)):
                x1, y1, d1, d2 = matches[i]
                x2, y2, f1, f2 = matches[j]

                if (x1 == x2 and y1 == y2) or (d1 == f1 and d2 == f2):
                    duplicate_indices.append(j)

    refined = []
    for i in range(len(matches)):
        if i not in duplicate_indices:
            refined.append(matches[i])

    return refined



# matches should be list of point matches
# [(x1,y1,x2,y2), ..., (xn,yn,xm,ym)]
# returns refined match list, fundamental matrix
def get_inliers_ransac(matches):
    max_iter = 900
    epsilon = 0.01
    num_matches = len(matches)
    if num_matches == 0:
        return [], None

    inlier_index_set = []
    optimal_fund_mat = None 

    for _ in tqdm(range(max_iter)):
        iter_inlier_index_set = []

        random_source_points = []
        random_dest_points = []        
        points_are_valid = False
        # repeating this to handle one-to-many points in matching
        while not points_are_valid:
            random_indices = np.random.choice(num_matches, 8, replace=False)
            random_source_points = []
            random_dest_points = []
            for r_index in random_indices:
                x1, y1, x2, y2 = matches[r_index]
                random_source_points.append((x1, y1))
                random_dest_points.append((x2, y2))

            duplicates_exists = False
            for i in range(7):
                for j in range(i+1, 8):
                    x1, y1 = random_source_points[i]
                    x2, y2 = random_source_points[j]
                    if x1 == x2 and y1 == y2:
                        duplicates_exists = True
            if not duplicates_exists:
                points_are_valid = True
        
        fund_mat = computeFundamentalMatrix(random_source_points, random_dest_points)
        if fund_mat is None:
            print(random_source_points, random_dest_points)
            continue

        average_error = 0.0


        for i in range(num_matches):
            x1, y1, x2, y2 = matches[i]
            vect2 = np.array([x2, y2, 1]).astype(np.float32)
            vect1 = np.array([x1, y1, 1]).astype(np.float32)
            
            try:
                fund_mat_error = abs(np.dot(np.dot(vect2, fund_mat), vect1))
            except:
                print(vect2, fund_mat, vect1)
                fund_mat_error = 10000000
            average_error += fund_mat_error
            if fund_mat_error < epsilon:
                iter_inlier_index_set.append(i)

        average_error = average_error / num_matches

        if len(iter_inlier_index_set) > len(inlier_index_set):
            inlier_index_set = iter_inlier_index_set.copy()
            optimal_fund_mat = fund_mat.copy()

    # compute refined matches from inlier index set
    refined_matches = []
    for index in inlier_index_set:
        refined_matches.append(matches[index])

    refined_matches = duplicate_refinement(refined_matches)

    return refined_matches, optimal_fund_mat
