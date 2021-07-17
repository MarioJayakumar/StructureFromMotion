import numpy as np

def buildVisibilityMatrix(correspondence, num_points):
    vis_mat = np.zeros((len(correspondence), num_points))

    for i in range(len(correspondence)): # iterating over i images...
        for corres in correspondence[i]:
            # corres is a dict mapping 2dPoint to 3d point index
            point_index = list(corres.values())[0]
            vis_mat[i][point_index] = 1

    vis_mat = vis_mat.astype(np.float32)
    return vis_mat