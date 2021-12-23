from Code.NonlinearPnP import nonlinearPnP
import cv2
from Code.GetInliersRANSAC import get_inliers_ransac
from Code.EstimateFundamentalMatrix import computeFundamentalMatrix
from Code.EssentialMatrixFromFundamentalMatrix import computeEssentialMatrixFromFundamentalMatrix
from Code.ExtractCameraPose import extractCameraPose
from Code.LinearTriangulation import linearTriangulation
from Code.DisambiguateCameraPose import disambiguateCameraPose
from Code.NonlinearTriangulation import nonLinearTriangulation
from Code.PnPRANSAC import computeReprojectionError, pnpRANSAC
from Code.BuildVisibilityMatrix import buildVisibilityMatrix
from Code.BundleAdjustment import bundleAdjustment
import numpy as np
import matplotlib.pyplot as plt

USE_RANSAC_CACHED =  True
USE_NONLINEARTRIANGULATION_CACHE = True

def compute_reprojection_error(correspondences, points_3d, C_set, R_set, K_mat):
    num_cameras = len(correspondences)
    average_error = 0.0
    count = 0
    for cam_index in range(num_cameras):
        C_vect = np.array(C_set[cam_index]).reshape((3,1))
        R_mat = R_set[cam_index]
        P_mat = np.matmul(np.matmul(K_mat, R_mat).astype(np.float32), np.hstack((np.eye(3), -1*C_vect)).astype(np.float32)).astype(np.float32)

        cam_correspondences = correspondences[cam_index]
        for corres in cam_correspondences:
            source_pair = list(corres.keys())[0]
            dest_index = list(corres.values())[0]

            u,v = source_pair
            X, Y, Z = points_3d[dest_index]
            homog_3d = np.array([X, Y, Z, 1]).astype(np.float32)

            term_left = u - (np.dot(P_mat[0], homog_3d))/(np.dot(P_mat[2], homog_3d))
            term_right = v - (np.dot(P_mat[1], homog_3d))/(np.dot(P_mat[2], homog_3d))
            local_error = term_left**2 + term_right**2
            average_error += local_error
            count += 1
    average_error = average_error / count
    print("Reproj Error", average_error)
    

def plot_ambiguous_poses(X_set, C_set, save=False, save_name=""):
    colors = ["b", "r", "g", "c", "m", "y", "brown"]
    x_data = []
    y_data = []
    color_data = []
    c_index = 0
    s_data = []
    for x_list in X_set:
        list_color = colors[c_index]
        c_index = (c_index + 1) % len(colors)
        for x_vect in x_list:
            # we are plotting x and z
            x_data.append(x_vect[0])
            y_data.append(x_vect[2])
            color_data.append(list_color)
            s_data.append(2)

    for i in range(len(C_set)):
        C_vect = C_set[i]
        x_data.append(C_vect[0])
        y_data.append(C_vect[2])
        color_data.append("black")
        s_data.append(4)
    plt.scatter(x_data, y_data, c=color_data, s=s_data)
    if not save:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.clf()

def draw_matching(left_img, right_img, matching, fname=""):
    left_keypoints = []
    right_keypoints = []
    for match in matching:
        a, b, c, d = match
        left_keypoints.append((int(a), int(b)))
        right_keypoints.append((int(c), int(d)))

    composite_img = np.concatenate((left_img, right_img), axis=1)
    h_cut = left_img.shape[1] # horizontal width of left image
    for l_key in left_keypoints:
        y, x = l_key
        cv2.circle(composite_img, (x,y), 2, (0, 255, 255), thickness=3)
    for r_key in right_keypoints:
        y, x = r_key
        cv2.circle(composite_img, (x+h_cut,  y), 2, (0, 255, 255), thickness=3)
    for k_index in range(len(left_keypoints)):
        y_orig, x_orig = left_keypoints[k_index]
        y_term, x_term = right_keypoints[k_index]
        cv2.line(composite_img, (x_orig, y_orig), (x_term+h_cut, y_term), color=(255, 255, 0), thickness=1)
    cv2.imwrite(fname, composite_img)      

if __name__ == "__main__":

    data_location = "Data/Imgs/"

    image_set = []
    matches = []
    K_mat = [[] ,[] ,[]]

    # initialize matches datastructure
    for i in range(6):
        matches.append([])
        for j in range(6):
            matches[i].append([])

    # load data
    for i in range(1,7):
        image_set.append(cv2.imread(data_location+str(i)+".jpg"))
        if i < 6:
            this_image_id = i-1
            with open(data_location+"matching"+str(i)+".txt", "r") as match_file_fh:
                num_line = match_file_fh.readline()
                num_line_split = num_line.split(":")
                num_features = int(num_line_split[1])
                for _ in range(num_features):
                    match_line = match_file_fh.readline()
                    match_line_split = match_line.split(" ")
                    num_matches = int(match_line_split[0]) - 1
                    source_y = float(match_line_split[4])
                    source_x = float(match_line_split[5])

                    other_match_data = match_line_split[6:]
                    for k in range(num_matches):
                        other_image_id = int(other_match_data[3*k]) - 1
                        other_image_y = float(other_match_data[3*k+1])
                        other_image_x = float(other_match_data[3*k+2])

                        matches[this_image_id][other_image_id].append((source_x, source_y, other_image_x, other_image_y))
                        #matches[other_image_id][this_image_id].append((other_image_x, other_image_y, source_x, source_y))
                

    with open(data_location+"calibration.txt", "r") as calib_fh:
        first_line = calib_fh.readline()
        first_row = first_line.split("[")[1]
        first_row = first_row[:-2]
        first_row = first_row.split(" ")
        second_line = calib_fh.readline().lstrip()
        second_row = second_line[:-2].split(" ")
        third_line = calib_fh.readline().lstrip()
        third_row = third_line[:-1].split(" ")        
        for i in range(3):
            K_mat[0].append(float(first_row[i]))      
            K_mat[1].append(float(second_row[i]))
            K_mat[2].append(float(third_row[i]))
    K_mat = np.array(K_mat).astype(np.float32)


    # compute RANSAC over all matches
    refined_matches = []
    fund_mat_set = []
    for i in range(6):
        refined_matches.append([])
        fund_mat_set.append([])
        for j in range(6):
            refined_matches[i].append([])    
            fund_mat_set[i].append([])

    if USE_RANSAC_CACHED:
        for i in range(5):
            #fund_mat_set.append([[]])
            #refined_matches.append([[]])
            for j in range(i+1, 6):
                cache_out_ransac_name = str(i) + "_" + str(j) + "_RANSAC.npy"
                cache_out_fund_name = str(i) + "_" + str(j) + "_fund.npy"

                inlier_set = np.load(cache_out_ransac_name, allow_pickle=True)
                fund_mat = np.load(cache_out_fund_name, allow_pickle=True)

                refined_matches[i][j] = inlier_set
                #fund_mat_set[i][j] = fund_mat                   
    else:
        for i in range(5):
            #fund_mat_set.append([[]])
            for j in range(i+1,6):
                local_matches = matches[i][j]
                inlier_set, fund_mat = get_inliers_ransac(local_matches)
                fund_mat_set[i][j] = fund_mat
                refined_matches[i][j] = inlier_set

                print(str(i) + " to " + str(j) + " refined points:", len(inlier_set))

                cache_out_ransac_name = str(i) + "_" + str(j) + "_RANSAC.npy"
                cache_out_fund_name = str(i) + "_" + str(j) + "_fund.npy"

                inlier_set = np.array(inlier_set)
                fund_mat = np.array(fund_mat)

                np.save(cache_out_ransac_name, inlier_set, allow_pickle=True)
                np.save(cache_out_fund_name, fund_mat, allow_pickle=True)

    # recompute fund mats using entire new refined matches
    for i in range(5):
        for j in range(i+1, 6):
            local_matches = refined_matches[i][j]
            src_points = []
            dst_points = []
            for sub_loc in local_matches:
                a, b, c, d = sub_loc
                src_points.append((a, b))
                dst_points.append((c, d))
            if len(src_points) > 8:
                fund_mat_set[i][j] = computeFundamentalMatrix(src_points, dst_points, src_image=image_set[i], dst_image=image_set[j])

                # evaluate error of this funamental matrix
                average_fund_mat_error = 0.0
                for k in range(len(local_matches)):
                    x1, y1, x2, y2 = local_matches[k]
                    vect2 = np.array([x2, y2, 1]).astype(np.float32).T
                    vect1 = np.array([x1, y1, 1]).astype(np.float32)
            
                    fund_mat_error = abs(np.dot(np.matmul(vect2, fund_mat), vect1))
                    average_fund_mat_error += fund_mat_error
                average_fund_mat_error = average_fund_mat_error / len(local_matches)
                print("Matching", i, "to", j, "F error:", average_fund_mat_error)
            else:
                fund_mat_set[i][j] = None

    draw_matching(image_set[0], image_set[1], matches[0][1], fname="unrefined_matching.png")
    draw_matching(image_set[0], image_set[1], refined_matches[0][1], fname="refined_matching.png")
    print(fund_mat_set[0][1])

    # compute essential matrix
    first_E_mat = computeEssentialMatrixFromFundamentalMatrix(fund_mat_set[0][1], K_mat) # probably correct 
    print("E:")
    print(first_E_mat)

    # extract pose of second camera
    C_set, R_set = extractCameraPose(first_E_mat) # verifiably correct by cv2 equivalent function

    for i in range(len(C_set)):
        print(i, C_set[i])
        print(R_set[i])
    
    # triangulate 3d points of points in second image
    X_set = []
    for i in range(4):
        X_set.append(linearTriangulation(K_mat, np.zeros((3, 1)), np.eye(3), C_set[i].reshape((3,1)), R_set[i], refined_matches[0][1]))

    plot_ambiguous_poses(X_set, C_set, save=True, save_name="MultiplePose.jpg")

    # perform cheirality test
    uniqueC, uniqueR, uniqueX = disambiguateCameraPose(C_set, R_set, X_set)

    plot_ambiguous_poses([uniqueX], [np.zeros((3,1))], save=True, save_name="UnAmbiguous.jpg")

    C_set = [np.zeros((3,1)), uniqueC]
    R_set = [np.eye(3), uniqueR]

    # element i contains correspondences for image i
    # correspondence contains mapping between points in an image and the 3d points we have mapped
    # we can start off with the triangulated results
    point_correspondence_list = []
    i0correspondence = []
    i1correspondence = []
    for j in range(len(uniqueX)):
        s_x, s_y, d_x, d_y = refined_matches[0][1][j]
        #X_x, X_y, X_z = uniqueX[j]
        i0correspondence.append({(s_x, s_y): j})
        i1correspondence.append({(d_x, d_y): j})
    point_correspondence_list.append(i0correspondence)
    point_correspondence_list.append(i1correspondence)

    print("Reproj error after linear triangulation")
    compute_reprojection_error(point_correspondence_list, uniqueX, C_set, R_set, K_mat)

    # further refine triangulated points
    optimized_points = None
    if USE_NONLINEARTRIANGULATION_CACHE:
        optimized_points = np.load("nonlineartriangulationoutput.npy")
    else:
        optimized_points = nonLinearTriangulation(K_mat, np.zeros((3, 1)), np.eye(3), uniqueC.reshape((3,1)), uniqueR, refined_matches[0][1], uniqueX)

        nonLinearTriangulationOut = "nonlineartriangulationoutput.npy"
        np.save(nonLinearTriangulationOut, optimized_points, allow_pickle=True)

    #plot_ambiguous_poses([uniqueX, optimized_points], [np.zeros((3,1)), np.zeros((3,1))])

    X_set = uniqueX

    print("Reproj error after nonlinear triangulation")
    compute_reprojection_error(point_correspondence_list, X_set, C_set, R_set, K_mat)

    # registering points in each additional image
    for i in range(2, len(image_set)):
        most_matching_image = -1
        maximal_matches = 0

        # find which image has most matches with image i
        for j in range(0, i):
            these_matches = len(refined_matches[j][i])
            if these_matches > maximal_matches:
                maximal_matches = these_matches
                most_matching_image = j

        print("Most matching to", i, "is", most_matching_image)
        C0 = C_set[most_matching_image]
        R0 = R_set[most_matching_image]

        # using the matching, find the 3d points that most relate to the points in image i
        closest_3D = []
        input_2d = []
        local_corresp_list = point_correspondence_list[most_matching_image]
        for j in range(len(refined_matches[most_matching_image][i])):
            s1, s2, d1, d2 = refined_matches[most_matching_image][i][j]

            # search for s1, s2 in correspondence list
            for cosp in local_corresp_list:
                if (s1, s2) in cosp:
                    index_3d = cosp[(s1, s2)]
                    closest_3D.append(X_set[index_3d])
                    input_2d.append((s1, s2))

        print("PnP RANSAC on", i)
        C_new, R_new = pnpRANSAC(closest_3D, input_2d, K_mat)
        print("nonlinear PnP Optim on", i)
        C_new, R_new = nonlinearPnP(closest_3D, input_2d, C_new, R_new, K_mat)

        # add new poses to list
        C_set.append(C_new)
        R_set.append(R_new)

        print("Triangulate points between", most_matching_image, "and",  i)
        new_X_set = linearTriangulation(K_mat, C0, R0, C_new, R_new, refined_matches[most_matching_image][i])
        new_X_set = nonLinearTriangulation(K_mat, C0, R0, C_new, R_new, refined_matches[most_matching_image][i], new_X_set)

        # need to add new X_vects to X_set. Register new ones in point_correspondences
        new_correspondences = []
        for new_X_index in range(len(new_X_set)):
            new_X = new_X_set[new_X_index]
            contained_in = False
            existing_ref_index = -1

            # check if new vect is actually alreading in the set
            for existing_X_vect_index in range(len(X_set)):
                existing_X_vect = X_set[existing_X_vect_index]
                if existing_X_vect[0] == new_X[0] and existing_X_vect[1] == new_X[1] and existing_X_vect[2] == new_X[2]:
                    contained_in = True
                    ref_index = existing_X_vect_index

            _, _, corr_x, corr_y = refined_matches[most_matching_image][i][new_X_index]
            if contained_in:
                index_track = len(X_set)
                X_set.append(new_X)
                new_correspondences.append({(corr_x, corr_y):index_track})
            else:
                new_correspondences.append({(corr_x, corr_y):existing_ref_index})
        point_correspondence_list.append(new_correspondences)

        print("Reproj error after PnP of", i, "via image", most_matching_image)
        compute_reprojection_error(point_correspondence_list, X_set, C_set, R_set, K_mat)

        print("Bundle adjustment with", i)
        vis_mat = buildVisibilityMatrix(point_correspondence_list, len(X_set))
        C_set, R_set, X_set = bundleAdjustment(C_set, R_set, X_set, point_correspondence_list, vis_mat, K_mat)

        print("Reproj error after bundle adjustment of", i)
        compute_reprojection_error(point_correspondence_list, X_set, C_set, R_set, K_mat)

    graphical_X_set = []
    for cam_index in range(len(point_correspondence_list)):
        local_X_set = []
        for corres in point_correspondence_list[cam_index]:
            dest_index = list(corres.values())[0]
            local_X_set.append(X_set[dest_index])
        graphical_X_set.append(local_X_set)
    plot_ambiguous_poses(graphical_X_set, C_set, save=True, save_name="FinalOutput.jpg")