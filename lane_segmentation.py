import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects


def _compute_x_values(mask, min_objects=1, x_dist_thresh=5):

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
                    if abs(x0 - _x0) < x_dist_thresh:
                        colineal_idxs.append(_idx)
                        colineal_x0s = [_x0]
            for colineal_idx in colineal_idxs:
                props_idx.remove(colineal_idx)

            if len(colineal_idxs) >= min_objects:
                colineal_x = int(np.mean(colineal_x0s))
                x_value.append(colineal_x)

        return x_value

def _filter_x_values(x_values_1, x_values_2):

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


    mask_1 = (mask == 1)
    mask_1 = binary_fill_holes(mask_1.astype(int)).astype(int)
    mask_1 = (remove_small_objects(measure.label(mask_1, connectivity=2), min_size=100) > 0)
    x_values_1 = _compute_x_values(mask_1, min_objects=2)
    x_values_1 = sorted(x_values_1)

    #figures figura fases segmentacion
    # mask_figure = np.zeros(mask.shape,dtype=np.uint8)
    # mask_figure[mask_1] = 1
    # for _x_value in x_values_1:
    #     mask_figure[:, _x_value]= 2
    # plt.figure()
    # plt.imshow(mask_figure)


    mask_2 = (mask == 2)
    mask_2 = binary_fill_holes(mask_2.astype(int)).astype(int)
    mask_2 = (remove_small_objects(measure.label(mask_2, connectivity=2), min_size=50) > 0)
    x_values_2 = _compute_x_values(mask_2)
    x_values_2 = sorted(x_values_2)

    #figures figura fases segmentacion
    # mask_figure1 = np.zeros(mask.shape,dtype=np.uint8)
    # mask_figure1[mask_2] = 1
    # for _x_value in x_values_2:
    #     mask_figure1[:, _x_value]= 2
    # plt.figure()
    # plt.imshow(mask_figure1)

    lane_limits = _filter_x_values(x_values_1, x_values_2)

    #figures figura fases segmentacion
    # mask_figure2 = np.zeros(mask.shape,dtype=np.uint8)
    # mask_figure2[mask_1] = 1
    # mask_figure2[mask_2] = 2
    # mask_figure2[:, lane_limits[1]] = 3
    # for _x_value in lane_limits[1:-1]:
    #     mask_figure2[:, _x_value]= 4
    # mask_figure2[:, lane_limits[-1]] = 3
    # plt.figure()
    # plt.imshow(mask_figure2)
    # plt.show()


    lanes_mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(len(lane_limits)-1):
        lanes_mask[:,lane_limits[i]:lane_limits[i+1]] = (i + 1)

    num_lanes = len(lane_limits)-1

    return lanes_mask, num_lanes
