import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage import measure
from scipy.spatial.distance import cdist
import math

def _get_line_coeffs(point, orientation):
    """
       Compute the coefficients of a line given a point and the perpendicular direction
       Args:
           point:                  point lying on the line
           orientation:            perpendicular direction to the line
       Return:
           line_coeffs:            [A,B,C] coefficients of the general equation of a line
    """
    x, y = point
    A = math.sin(orientation)
    B = math.cos(orientation)
    C = -(A * x + B * y)
    return (A, B, C)

def _get_dist_to_line(line_coeffs, point):
    """
       Compute the min distance to a line from a point
       Args:
           line_coeffs:            [A,B,C] coefficients of the general equation of a line
           point:                  point from which to measure distance to line
       Return:
           coeffs:                 [A,B,C] coefficients of the general equation of a line
    """
    x, y = point
    A, B, C = line_coeffs
    dist = abs(A * x + B * y + C)/math.sqrt((A ** 2) + (B ** 2))
    return dist


def _get_colineal_centroids(prop_idx, props, colineal_thresh=15, distance_thresh=350):

    """
       Compute all the colinear centroids to an anchor centroid
       Args:
           prop_idx:               index of the anchor centroid in the props array
           props:                  array of props obtained from skimage.measure.regionprops
           colineal_thresh:        maximum distance to the orientation line for a centroid to be considered colineal
           distance_thresh:        maximum distance to the anchor centroid to be considered colineal

       Return:
           colineal_centroids:     list of indices in the props array of the colineal centroids
           distances:              list of distances of the distances to the anchor centroid
    """

    y0, x0 = props[prop_idx]["centroid"]
    orientation = props[prop_idx]["orientation"]
    line_coeffs = _get_line_coeffs((x0, y0), orientation)

    colineal_centroids = []
    distances = []

    for _prop_idx in range(len(props)):
        if _prop_idx != prop_idx:
            _y0, _x0 = props[_prop_idx]["centroid"]
            if _get_dist_to_line(line_coeffs, (_x0, _y0)) < colineal_thresh:
                distances.append(cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)])))
                if distances[-1] < distance_thresh:
                    colineal_centroids.append(_prop_idx)
                else:
                    distances.pop()

    return colineal_centroids, distances

def _filter_centroids(props, prop_idx, colineal_points, distances):
    """
       Filter the colinear centroids so that only remain the adyacent ones
       Args:
           props:                  array of props obtained from skimage.measure.regionprops
           prop_idx:               index of the anchor centroid in the props array
           colineal_points:        list of indices in the props array of the colineal centroids
           distances:              maximum distance to the anchor centroid to be considered colineal

       Return:
           adjacent_centroids:     list of indices of the adyacent centroids to the anchor centroid
    """

    adjacent_centroids = []

    if colineal_points:

        colineal_points, distances = (list(t) for t in zip(*sorted(zip(colineal_points, distances), key = lambda x: x[1])))
        adjacent_centroids.append(colineal_points.pop(0))

        y1, x1 = props[adjacent_centroids[0]]["centroid"]
        y0, x0 = props[prop_idx]["centroid"]

        while (len(adjacent_centroids) != 2) and (len(colineal_points) > 0):
            y2, x2 = props[colineal_points[0]]["centroid"]
            dist_2_0 = cdist(np.array([(x0,y0)]), np.array([(x2,y2)]))
            dist_2_1 = cdist(np.array([(x1,y1)]), np.array([(x2,y2)]))
            if dist_2_1 > dist_2_0:
                adjacent_centroids.append(colineal_points.pop(0))
            else:
                colineal_points.pop(0)

    return adjacent_centroids

def _get_centroids_groups(adjacent_centroids_list):
    """
       Walks the adyacent_centroids_list starting from the endpoints to obtain the centroid groups
       Args:
           adjacent_centroids_list:     list of tuples with the indices of the adyacent centroids of each centroid
       Return:
           centroid_groups:             list of groups of centroids, havings each group two endpoints
    """

    endpoints = []
    for idx, adjacent_centroids in enumerate(adjacent_centroids_list):
        if len(adjacent_centroids) == 1:
            endpoints.append(idx)

    centroid_groups = []
    while endpoints:

        centroids_idx = []
        endpoint_idx = endpoints.pop(0)
        centroids_idx.append(endpoint_idx)
        current_idx = endpoint_idx
        next_idx = adjacent_centroids_list[current_idx][0]
        lane_completed = False
        
        while not lane_completed:

            centroids_idx.append(next_idx)
            _next_idx = list(set(adjacent_centroids_list[next_idx]) - set([current_idx]))
            
            if _next_idx:
                current_idx = next_idx
                next_idx = _next_idx[0]
            else:
                print
                endpoints.remove(next_idx)
                lane_completed = True

        if len(centroids_idx) > 2:
            centroid_groups.append(centroids_idx)

    return centroid_groups


def group_marks(total_mask):
    """
       Groups the inner lines so that each group is a lane limit
       Args:
           total_mask:                  bool mask of inner lines on the aerial image coordinate system
       Return:
           new_total_mask:              integer mask where each integer is associated to a group
           num_groups:                  number of groups in the mask
    """

    props = measure.regionprops(measure.label(total_mask, connectivity=2), coordinates='xy')
    adjacent_centroids_list = []

    for prop_idx in range(len(props)):

        colineal_centroids, distances = _get_colineal_centroids(prop_idx, props)
        adjacent_centroids = _filter_centroids(props, prop_idx, colineal_centroids, distances)
        adjacent_centroids_list.append(adjacent_centroids)

    centroid_groups = _get_centroids_groups(adjacent_centroids_list)

    total_mask_labeled = measure.label(total_mask, connectivity=2)
    new_total_mask = np.zeros(total_mask.shape, dtype=np.uint8)
    num_groups = len(centroid_groups)

    for group_idx in range(num_groups):
        for mark_idx in centroid_groups[group_idx]:
            new_total_mask[total_mask_labeled == props[mark_idx]["label"]] = (group_idx + 1)

    return new_total_mask, num_groups