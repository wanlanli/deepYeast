import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture


def find_nearnest_points(x, y):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
    distances, index = nbrs.kneighbors(x)
    dis = np.min(distances)
    ind = np.argmin(distances)
    return dis, ind, index[ind][0]


# def predition_data_type(data, **args):
#     clustering = GaussianMixture(**args).fit(data)
#     data_pred = clustering.predict(data)
#     class_map = rename_classes(data, clustering.means_)
#     data_pred = [class_map[x] for x in data_pred]
#     return data_pred, clustering


# def _get_bounding_points(data): 
#     box_min = data.min()
#     box_max = data.max()
#     coords = np.array([box_min, box_max]).T
#     import itertools
#     coords_index = np.array(list(itertools.product([0, 1], repeat=coords.shape[0])))
#     out = []
#     for i in range(coords.shape[0]):
#         out.append(coords[i, coords_index[:, i]])
#     out = np.array(out).T
#     return out


# def rename_classes(data, cluster_centers):
#     coords = _get_bounding_points(data)
#     class_map = {}
#     for i, c in enumerate(cluster_centers):
#         dist = np.sqrt(np.square(coords - c).sum(axis=1))
#         label = np.argmin(dist)
#         class_map[i] = label
#     return class_map
