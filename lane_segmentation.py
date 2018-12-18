import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects


def _compute_x_values(mask, min_objects=1, x_dist_thresh=5):
	"""
       Compute vertical lines closest to each group of colineal centroids
       Args:
           mask:             boolean mask
           min_objects:      min number of colineal centroids
           x_dist_thresh:    max distance between centroids to be considered colineal
       Return:
           x_value:          list of x_value for each group of colineal centroids
    """
	props = measure.regionprops(measure.label(mask, connectivity=2), coordinates='xy')
	props_idx = range(len(props))
	x_value = []
	while props_idx:

	    idx = props_idx[0]
	    y0, x0 = props[idx]["centroid"]
	    colineal_idxs = [idx]
	    colineal_x0s = [x0]

	    for _idx in props_idx:
	        if _idx != idx:
	            _y0, _x0 = props[_idx]["centroid"]
	            print abs(x0 - _x0)
	            if abs(x0 - _x0) < x_dist_thresh:
	                colineal_idxs.append(_idx)
	                colineal_x0s = [_x0]
	        print 
	    for colineal_idx in colineal_idxs:
	        props_idx.remove(colineal_idx)

	    if len(colineal_idxs) >= min_objects:
	        colineal_x = int(np.mean(colineal_x0s))
	        x_value.append(colineal_x)

	return x_value

def _filter_x_values(x_values_1, x_values_2):

	"""
       Filter colineal groups so that only remain the ones inside the road being segmented
       Args:
           x_values_1:       list of colineal groups of inner lines
           x_values_2:       list of colineal groups of outter lines
       Return:
           lane_limits:      lane_limits for the road being segmented
    """

	lane_limits = None
	for i in range(len(x_values_2)-1):
		x_value_1_selected = []
		for x_value_1 in x_values_1:
			if x_values_2[i] < x_value_1 < x_values_2[i+1]:
				x_value_1_selected.append(x_value_1)
		if x_value_1_selected:
			lane_limits = x_value_1_selected
			lane_limits.insert(0, x_values_2[i])
			lane_limits.append(x_values_2[i+1])
			break
	
	return lane_limits

def segment_lanes(mask):
	"""
	   Filter colineal groups so that only remain the ones inside the road being segmented
	   Args:
	       mask:            integer mask with classes 0,1,2,3,4
	   Return:
	       lanes_mask:      integer mask of the lanes
	       num_lanes:       number of lanes segmented
	"""
	mask_1 = (mask == 1)
	mask_1 = binary_fill_holes(mask_1.astype(int)).astype(int)
	mask_1 = (remove_small_objects(measure.label(mask_1, connectivity=2), min_size=100) > 0)
	x_values_1 = _compute_x_values(mask_1, min_objects=2)
	x_values_1 = sorted(x_values_1)

	mask_2 = (mask == 2)
	mask_2 = binary_fill_holes(mask_2.astype(int)).astype(int)
	mask_2 = (remove_small_objects(measure.label(mask_2, connectivity=2), min_size=50) > 0)
	x_values_2 = _compute_x_values(mask_2)
	x_values_2 = sorted(x_values_2)

	lane_limits = _filter_x_values(x_values_1, x_values_2)

	lanes_mask = np.zeros(mask.shape, dtype=np.uint8)
	for i in range(len(lane_limits)-1):
	    lanes_mask[:,lane_limits[i]:lane_limits[i+1]] = (i + 1)

	num_lanes = len(lane_limits)-1

	return lanes_mask, num_lanes
