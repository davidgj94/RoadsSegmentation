import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage import measure
from scipy.spatial.distance import cdist
import math

def _get_line_coeffs(point, orientation):
    x, y = point
    A = math.sin(orientation)
    B = math.cos(orientation)
    C = -(A * x + B * y)
    return (A, B, C)

def _get_dist_to_line(line_coeffs, point):
    x, y = point
    A, B, C = line_coeffs
    dist = abs(A * x + B * y + C)/math.sqrt((A ** 2) + (B ** 2))
    return dist


def _get_colineal_centroids(prop_idx, props, colineal_thresh=15, distance_thresh=350):

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

    print adjacent_centroids_list

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
        print current_idx
        while not lane_completed:

            centroids_idx.append(next_idx)
            _next_idx = list(set(adjacent_centroids_list[next_idx]) - set([current_idx]))
            print next_idx
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


def group_marks(total_mask, colineal_thresh=15, distance_thresh=350):

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