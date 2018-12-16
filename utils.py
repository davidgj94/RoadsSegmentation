import numpy as np
import skimage.io
from PIL import Image
import urllib2
import io
from skimage import exposure
from urllib import pathname2url as quote
import sys
import matplotlib.pyplot as plt
import pdb
from skimage import morphology
import networkx as nx
import cv2
from matplotlib.patches import Circle
import mahotas as mah
from skimage import measure
from scipy.spatial.distance import cdist
import math
from math import ceil
from scipy.ndimage.morphology import binary_fill_holes
import vis
from skimage.morphology import remove_small_objects
from math import ceil
import tensorflow as tf
import segnet_TF
from PIL import Image, ImageDraw

# caffe_root = "/home/davidgj/projects_v2/caffe-segnet-cudnn5/"

# sys.path.insert(0, caffe_root + 'python')

# import caffe; caffe.set_mode_gpu()

# prototxt = "/home/davidgj/projects_v2/SegNet-Tutorial/Models/roads/inference_video.prototxt"
# caffemodel = "/home/davidgj/projects_v2/SegNet-Tutorial/Models/Inference/roads/snapshot_iter_1500/test_weights.caffemodel"
# net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# def road_segmentation(img):

#     img = img[...,::-1]
#     img = img.transpose((2,0,1))
#     img = img[np.newaxis,...]
#     print img.shape

#     net.blobs['data'].reshape(*img.shape)
#     net.blobs['data'].data[...] = img

#     net.forward()

#     predicted = np.squeeze(net.blobs['prob'].data)
#     ind = np.argmax(predicted, axis=0)

#     return ind

base_URL = "https://maps.googleapis.com/maps/api/staticmap?key=AIzaSyDvgF0JSBrlYLDzY7pPqtcBSgGslmaAlzw&zoom=19&format=png&maptype=roadmap&style=color:0x000000&style=element:labels%7Cvisibility:off&style=feature:road%7Celement:geometry%7Ccolor:0xffffff%7Cvisibility:on&style=feature:road.highway%7Celement:geometry%7Ccolor:0xffffff%7Cvisibility:on&style=feature:road.local%7Celement:geometry%7Cvisibility:off&size=640x640&scale=2"

satellite_URL = "https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&zoom=19&format=png&size=640x640&scale=2&key=AIzaSyDvgF0JSBrlYLDzY7pPqtcBSgGslmaAlzw"

sys.argv.pop(0)
coord = tuple(sys.argv)

def download_img_mask(coord):
    
    new_url = base_URL + "&center=" + quote("{}, {}".format(*coord))
    new_satellite_url = satellite_URL + "&center=" + quote("{}, {}".format(*coord))
    
    url = urllib2.urlopen(new_url)
    f = io.BytesIO(url.read())
    #pdb.set_trace()
    mask = exposure.rescale_intensity(np.array(Image.open(f)))
    mask[mask != 255] = 0
    img = skimage.io.imread(new_satellite_url)
    
    return img, mask

def _get_relav_dist(neighbor_coord):
    return (neighbor_coord[0] - 1, neighbor_coord[1] - 1)

def _get_angle(p1, p2):
    xx = (p1[1], p2[1])
    yy = (p1[0], p2[0])

    if xx[0] == xx[1]:
        angle = math.degrees(math.pi/2)
    else:
        a = float(xx[0] - xx[1])
        b = float(yy[0] - yy[1])
        angle = math.degrees(math.atan(b/a))
    return angle

def _get_midpoint(p1, p2):
    xx = (p1[1], p2[1])
    yy = (p1[0], p2[0])
    x_midpoint = (xx[0] + xx[1])/2
    y_midpoint = (yy[0] + yy[1])/2
    return y_midpoint, x_midpoint

def get_padding(sz, sz_):
    
    pad_amount = sz_ - sz
    
    if pad_amount % 2:
        padding = (pad_amount / 2 , pad_amount - pad_amount / 2)
    else:
        padding = (pad_amount / 2, pad_amount / 2)
        
    return padding

def pad_img(img, sz, pad_value=0):

    H, W = sz
    height, width = img.shape[:2]
    x_pad = get_padding(width, W)
    y_pad = get_padding(height, H)
    img_padded = cv2.copyMakeBorder(img, y_pad[0], y_pad[1], x_pad[0], x_pad[1], cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

    return img_padded, x_pad, y_pad

def get_line_coeffs(point, orientation):
    x, y = point
    A = math.sin(orientation)
    B = math.cos(orientation)
    C = -(A * x + B * y)
    return (A, B, C)

def get_dist_to_line(line_coeffs, point):
    x, y = point
    A, B, C = line_coeffs
    dist = abs(A * x + B * y + C)/math.sqrt((A ** 2) + (B ** 2))
    return dist

def find_intersect(line1_coeffs, line2_coeffs):

    A1, B1, C1 = line1_coeffs
    A2, B2, C2 = line2_coeffs

    denom = (A1 * B2 - B1 * A2)
    if abs(denom) > 1e-10:
        x = (B1 * C2 - C1 * B2) / denom
        y = (C1 * A2 - A1 * C2) / denom
    else:
        return None

    return (x, y)

def find_intesect_borders(line_coeffs, sz):

    def check_intersect(point, sz):
        H, W = sz
        return (0.0 <= point[0] <= float(W)) and (0.0 <= point[1] <= float(H))

    H, W = sz

    upper_border_coeffs = (0.0, 1.0, 0.0)
    lower_border_coeffs = (0.0, 1.0, -float(H))
    left_border_coeffs = (1.0, 0.0, 0.0)
    right_border_coeffs = (1.0, 0.0, -float(W))

    upper_border_intersect = find_intersect(line_coeffs, upper_border_coeffs)
    lower_border_intersect = find_intersect(line_coeffs, lower_border_coeffs)
    left_border_intersect = find_intersect(line_coeffs, left_border_coeffs)
    right_border_intersect = find_intersect(line_coeffs, right_border_coeffs)

    intersect_points = []
    if check_intersect(upper_border_intersect, sz):
        intersect_points.append(upper_border_intersect)
    if check_intersect(lower_border_intersect, sz):
        intersect_points.append(lower_border_intersect)
    if check_intersect(left_border_intersect, sz):
        intersect_points.append(left_border_intersect)
    if check_intersect(right_border_intersect, sz):
        intersect_points.append(right_border_intersect)

    return intersect_points

def divide_skel(coords, D=200, M=200, thresh=2, min_height=100):
    
    ref_index = 0
    angles = []
    sections_start = [coords[M]]
    section_mid_points = []
    section_angles = []
    section_height = []
    
    def _add_new_section(new_point):
        
        sections_start.append(coords[center_index])
        p1, p2 = sections_start[-2], sections_start[-1]
        
        height = int(cdist([p1], [p2]))
        if height > min_height:
            section_mid_points.append(_get_midpoint(p1, p2))
            section_angles.append(_get_angle(p1, p2))
            section_height.append(height)
        else:
            sections_start.pop()

    for center_index in range(M, (len(coords) - 1) - M, D):
        angle = _get_angle(coords[center_index - D], coords[center_index + D])
        angles.append(angle)
        if abs(angle - angles[ref_index]) > thresh:
            ref_index = len(angles) - 1
            _add_new_section(coords[center_index])
            
    _add_new_section(coords[center_index])
            
    return section_mid_points, section_angles, section_height, angles


def extract_section(img, center, angle, height, width=200):
    
    img_h, img_w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center[::-1], angle - 90, 1.0)
    rotated = cv2.warpAffine(img, M, (img_h, img_w), cv2.INTER_NEAREST)
    # circ1 = Circle(center[::-1],20)
    # circ2 = Circle(center[::-1],20)
    # circ3 = Circle(center[::-1],20)
    # fig1,ax1 = plt.subplots(1)
    # ax1.imshow(img)
    # ax1.add_patch(circ1)
    # fig2,ax2 = plt.subplots(1)
    # ax2.imshow(rotated)
    # ax2.add_patch(circ2)
    crop = cv2.getRectSubPix(rotated, (width, height), center[::-1])
    #crop = pad_img(crop)
    #pdb.set_trace()
    # fig1,ax3 = plt.subplots(1)
    # ax3.imshow(crop)
    
    # toltal_mask = paste_mask(np.zeros(img.shape).astype(bool), crop, center, angle)
    # ax3.imshow(toltal_mask)
    # ax3.add_patch(circ3)
    # plt.show()
    
    return crop

def paste_mask(total_mask, mask, center, angle):
    
    H, W = mask.shape
    y, x = np.where(mask)
    y += center[0] - H / 2
    x += center[1] - W / 2
    coords = np.array(zip(x, y, np.ones(x.shape, dtype=int)))
    M = cv2.getRotationMatrix2D(center[::-1], 90 - angle, 1.0)
    new_coords = M.dot(coords.T)
    total_mask[new_coords[1,:].astype(int), new_coords[0,:].astype(int)] = True
    return total_mask
    

def _get_next_point(skel, current_point, prev_point=None):
    
    upper_limit = current_point[0]-1
    bottom_limit = current_point[0]+1
    left_limit = current_point[1]-1
    right_limit = current_point[1]+1
    
    neighbors = skel[upper_limit:bottom_limit+1, left_limit:right_limit+1]
    y, x = np.where(neighbors)
    neighbors_coord = zip(y,x)

    if prev_point:
        relav_dist = _get_relav_dist(prev_point)
        prev_point = (1 - relav_dist[0], 1 - relav_dist[1])
        
    neighbors_coord = [point_coord for point_coord in neighbors_coord if point_coord not in [(1,1), prev_point]]
    
    next_point = neighbors_coord[np.argmin(cdist([(1,1)], neighbors_coord))]
    
    return next_point
    

def sort_skeleton(skel, img, min_num_points=500):

    br_mask = np.zeros(skel.shape, dtype=np.uint8)
    br, ep = detect_br_ep(skel)

    br_y, br_x = np.where(br)
    br_coords = zip(br_x, br_y)
    for br_x, br_y in br_coords:
        br_mask[br_y-1:(br_y+1)+1, br_x-1:(br_x+1)+1] = 1
    skel[br_mask.astype(bool)] = False

    # plt.figure()
    # plt.imshow(skel)
    # skel_labeled, num_labels = measure.label(skel, connectivity=2, return_num=True)
    # for i in range(num_labels):
    #     skel_cc = (skel_labeled == (i+1))
    #     plt.figure()
    #     plt.imshow(skel_cc)
    # plt.show()
    
    skel_labeled, num_labels = measure.label(skel, connectivity=2, return_num=True)

    sorted_coord_labels = []

    for i in range(num_labels):
        
        skel_cc = (skel_labeled == (i+1))

        
        if len(np.where(skel_cc)[0]) < min_num_points:
            continue
        
        br, ep = detect_br_ep(skel_cc)
        ep_y, ep_x = np.where(ep)
        br_y, br_x = np.where(br)
        ep_coords = zip(ep_y, ep_x)

        # ep_coords_ = zip(ep_x, ep_y)
        # br_coords = zip(br_x, br_y)

        # fig, ax = plt.subplots(1)
        # ax.imshow(skel_cc)
        # for ep_i in range(len(ep_coords_)):
        #     ax.add_patch(Circle(ep_coords_[ep_i], 10, color='k'))

        # for br_i in range(len(br_coords)):
        #     ax.add_patch(Circle(br_coords[br_i], 10, color='r'))

        # plt.show()
        
        sorted_coord = []
        sorted_coord.append(ep_coords[0])
        _next_point = _get_next_point(skel_cc, sorted_coord[-1])
        relav_dist = _get_relav_dist(_next_point)
        next_point = (sorted_coord[-1][0] + relav_dist[0], sorted_coord[-1][1] + relav_dist[1])
        sorted_coord.append(next_point)
        
        while next_point != ep_coords[1]:
             
            current_point = next_point
            _next_point = _get_next_point(skel_cc, current_point, _next_point)
            relav_dist = _get_relav_dist(_next_point)
            next_point = (current_point[0] + relav_dist[0], current_point[1] + relav_dist[1])
            sorted_coord.append(next_point)

        sorted_coord_labels.append(sorted_coord)

        # section_mid_points, section_angles, section_height, angles = divide_skel(sorted_coord)

    #     for index, _ in enumerate(section_mid_points):
    #         crop = extract_section(img, section_mid_points[index], section_angles[index], section_height[index])
    #         plt.figure()
    #         plt.imshow(crop)
    # plt.show()

        
        # total_mask = np.zeros(img.shape).astype(bool)
        # total_mask = total_mask[...,0]
        # for index, _ in enumerate(section_mid_points):
        #     crop = extract_section(img, section_mid_points[index], section_angles[index], section_height[index])
        #     crop, pad_x, pad_y = pad_img(crop, (384, 224))
        #     mask = road_segmentation(crop)
        #     mask = mask[pad_y[0]:384-pad_y[1], pad_x[0]:224-pad_x[1]]
        #     mask = (mask == 1)
        #     total_mask = paste_mask(total_mask, mask, section_mid_points[index], section_angles[index])
        #     # pdb.set_trace()
        # total_mask = binary_fill_holes(total_mask.astype(int)).astype(int)
        # total_mask = remove_small_objects(measure.label(total_mask, connectivity=2), min_size=100)
        # total_mask = (total_mask > 0).astype(np.uint8)


        # vis_img = vis.vis_seg(img, total_mask, np.array([[255,255,255],[0, 0, 255],[0, 255, 0]]))
        # plt.imshow(vis_img)
        # plt.show()
        # #pdb.set_trace()

        # # toltal_mask = binary_fill_holes(toltal_mask.astype(int)).astype(int)
        # # toltal_mask = remove_small_objects(measure.label(toltal_mask, connectivity=2), min_size=100)
        # # toltal_mask = (toltal_mask > 0).astype(np.uint8)
        # props = measure.regionprops(measure.label(total_mask, connectivity=2), coordinates='xy')

        # colineal_thresh = 15
        # distance_thresh = 350
        # colineal_points_total = []
        # from PIL import Image, ImageDraw

        # for prop_idx in range(len(props)):
        #     y0, x0 = props[prop_idx]["centroid"]
        #     orientation = props[prop_idx]["orientation"]
        #     line_coeffs = get_line_coeffs((x0, y0), orientation)
        #     # intersect_points = find_intesect_borders(line_coeffs, toltal_mask.shape)
        #     # im = Image.fromarray(toltal_mask)
        #     # draw = ImageDraw.Draw(im) 
        #     # draw.line([intersect_points[0], intersect_points[1]], fill=1, width=5)
        #     # plt.imshow(np.array(im))
        #     # plt.show()
        #     colineal_points = []
        #     distances = []
        #     for _prop_idx in range(len(props)):
        #         if _prop_idx != prop_idx:
        #             _y0, _x0 = props[_prop_idx]["centroid"]
        #             print get_dist_to_line(line_coeffs, (_x0, _y0))
        #             print cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)]))
        #             print _prop_idx

        #             if get_dist_to_line(line_coeffs, (_x0, _y0)) < colineal_thresh:
        #                 distances.append(cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)])))
        #                 if distances[-1] < distance_thresh:
        #                     colineal_points.append(_prop_idx)
        #                 else:
        #                     distances.pop()
    
        #     final_points = []

        #     colineal_points, distances = (list(t) for t in zip(*sorted(zip(colineal_points, distances), key = lambda x: x[1])))
        #     #pdb.set_trace()
        #     final_points.append(colineal_points.pop(0))
        #     y1, x1 = props[final_points[0]]["centroid"]

        #     while (len(final_points) != 2) and (len(colineal_points) > 0):
        #         y2, x2 = props[colineal_points[0]]["centroid"]
        #         dist_2_0 = cdist(np.array([(x0,y0)]), np.array([(x2,y2)]))
        #         dist_2_1 = cdist(np.array([(x1,y1)]), np.array([(x2,y2)]))
        #         if dist_2_1 > dist_2_0:
        #             final_points.append(colineal_points.pop(0))
        #         else:
        #             colineal_points.pop(0)


        #     colineal_points_total.append(final_points)

        # lane_endpoints = []
        # for idx, _colineal_point in enumerate(colineal_points_total):
        #     if len(_colineal_point) == 1:
        #         lane_endpoints.append(idx)

        # lanes = []
        # while lane_endpoints:

        #     marks_idx = []
        #     endpoint_idx = lane_endpoints.pop(0)
        #     marks_idx.append(endpoint_idx)
        #     current_idx = endpoint_idx
        #     next_idx = colineal_points_total[current_idx][0]
        #     lane_completed = False

        #     while not lane_completed:

        #         marks_idx.append(next_idx)
        #         _next_idx = list(set(colineal_points_total[next_idx]) - set([current_idx]))

        #         if _next_idx:
        #             current_idx = next_idx
        #             next_idx = _next_idx[0]
        #         else:
        #             lane_endpoints.remove(next_idx)
        #             lane_completed = True

        #     lanes.append(marks_idx)

        # total_mask_labeled = measure.label(total_mask, connectivity=2)
        # new_total_mask = np.zeros(total_mask.shape, dtype=np.uint8)
        # for lane_idx in range(len(lanes)):
        #     for mark_idx in lanes[lane_idx]:
        #         new_total_mask[total_mask_labeled == props[mark_idx]["label"]] = (lane_idx + 1)

        # plt.imshow(new_total_mask)
        # plt.show()




        # for prop_idx in range(len(props)):

        #     fig, ax = plt.subplots(1)
        #     ax.imshow(total_mask)

        #     y0, x0 = props[prop_idx]["centroid"]
        #     circle = Circle((x0, y0), 10, color='r')
        #     ax.add_patch(circle)

        #     for colineal_idx in range(len(colineal_points_total[prop_idx])):
        #         _prop_idx = colineal_points_total[prop_idx][colineal_idx]
        #         y0, x0 = props[_prop_idx]["centroid"]
        #         circle = Circle((x0, y0), 10, color='k')
        #         ax.add_patch(circle)

        #     plt.show()
        
    return sorted_coord_labels

def test_sort_skeleton():
    
    skel = np.array([[0,0,0,0,0],
                     [0,0,0,1,0],
                     [0,1,0,0,1],
                     [0,0,1,1,0],
                     [0,0,0,0,0]], dtype=bool)
    sorted_points = sort_skeleton(skel)
    pdb.set_trace()
    print "Fin"

def segment_marks(total_mask, colineal_thresh=15, distance_thresh=350):

    props = measure.regionprops(measure.label(total_mask, connectivity=2), coordinates='xy')
    colineal_points_total = []

    for prop_idx in range(len(props)):

        y0, x0 = props[prop_idx]["centroid"]
        orientation = props[prop_idx]["orientation"]
        line_coeffs = get_line_coeffs((x0, y0), orientation)

        colineal_points = []
        distances = []
        for _prop_idx in range(len(props)):
            if _prop_idx != prop_idx:
                _y0, _x0 = props[_prop_idx]["centroid"]
                # print get_dist_to_line(line_coeffs, (_x0, _y0))
                # print cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)]))
                # print _prop_idx
                if get_dist_to_line(line_coeffs, (_x0, _y0)) < colineal_thresh:
                    distances.append(cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)])))
                    if distances[-1] < distance_thresh:
                        colineal_points.append(_prop_idx)
                    else:
                        distances.pop()

        final_points = []
        colineal_points, distances = (list(t) for t in zip(*sorted(zip(colineal_points, distances), key = lambda x: x[1])))
        final_points.append(colineal_points.pop(0))
        y1, x1 = props[final_points[0]]["centroid"]

        while (len(final_points) != 2) and (len(colineal_points) > 0):
            y2, x2 = props[colineal_points[0]]["centroid"]
            dist_2_0 = cdist(np.array([(x0,y0)]), np.array([(x2,y2)]))
            dist_2_1 = cdist(np.array([(x1,y1)]), np.array([(x2,y2)]))
            if dist_2_1 > dist_2_0:
                final_points.append(colineal_points.pop(0))
            else:
                colineal_points.pop(0)


        colineal_points_total.append(final_points)

    lane_endpoints = []
    for idx, _colineal_point in enumerate(colineal_points_total):
        if len(_colineal_point) == 1:
            lane_endpoints.append(idx)

    lanes = []
    while lane_endpoints:

        marks_idx = []
        endpoint_idx = lane_endpoints.pop(0)
        marks_idx.append(endpoint_idx)
        current_idx = endpoint_idx
        next_idx = colineal_points_total[current_idx][0]
        lane_completed = False

        while not lane_completed:

            marks_idx.append(next_idx)
            _next_idx = list(set(colineal_points_total[next_idx]) - set([current_idx]))

            if _next_idx:
                current_idx = next_idx
                next_idx = _next_idx[0]
            else:
                lane_endpoints.remove(next_idx)
                lane_completed = True

        if len(marks_idx) > 2:
            lanes.append(marks_idx)

    total_mask_labeled = measure.label(total_mask, connectivity=2)
    new_total_mask = np.zeros(total_mask.shape, dtype=np.uint8)
    num_lanes = len(lanes)
    for lane_idx in range(num_lanes):
        for mark_idx in lanes[lane_idx]:
            new_total_mask[total_mask_labeled == props[mark_idx]["label"]] = (lane_idx + 1)

    return new_total_mask, num_lanes



def segment_one_section(mask, x_dist_thresh=5):


    def compute_x_values(_mask, min_objects=1):

        props = measure.regionprops(measure.label(_mask, connectivity=2), coordinates='xy')
        props_idx = range(len(props))
        x_value = []
        while props_idx:

            idx = props_idx[0]
            y0, x0 = props[idx]["centroid"]
            lane_idxs = [idx]
            lane_x0s = [x0]

            for _idx in props_idx:
                if _idx != idx:
                    _y0, _x0 = props[_idx]["centroid"]
                    print abs(x0 - _x0)
                    if abs(x0 - _x0) < x_dist_thresh:
                        lane_idxs.append(_idx)
                        lane_x0s = [_x0]
                print 
            for lane_idx in lane_idxs:
                props_idx.remove(lane_idx)

            if len(lane_idxs) >= min_objects:
                lane_x = int(np.mean(lane_x0s))
                x_value.append(lane_x)

        return x_value

    mask_1 = (mask == 1)
    mask_1 = binary_fill_holes(mask_1.astype(int)).astype(int)
    mask_1 = (remove_small_objects(measure.label(mask_1, connectivity=2), min_size=100) > 0)
    plt.figure()
    plt.imshow(mask_1)
    x_values_1 = sorted(compute_x_values(mask_1, min_objects=2))

    mask_2 = (mask == 2)
    mask_2 = binary_fill_holes(mask_2.astype(int)).astype(int)
    mask_2 = (remove_small_objects(measure.label(mask_2, connectivity=2), min_size=300) > 0)
    plt.figure()
    plt.imshow(mask_2)
    plt.show()
    x_values_2 = sorted(compute_x_values(mask_2))

    pdb.set_trace()

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

    # lane_limits = sorted(lanes)
    # lane_limits.insert(0, 0)
    # lane_limits.append(mask.shape[1])

    _mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(len(lane_limits)-1):
        _mask[:,lane_limits[i]:lane_limits[i+1]] = (i + 1)

    num_lanes = len(lane_limits)-1

    return _mask, num_lanes

    # plt.figure()
    # plt.imshow(mask)
    # plt.figure()
    # plt.imshow(_mask)
    # plt.show()






def detect_br_ep(sk):
    
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    
    endpoint1=np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
    endpoint2=np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    endpoint3=np.array([[0, 0, 2], [0, 1, 2], [0, 2, 1]])
    endpoint4=np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
    endpoint5=np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
    endpoint6=np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
    endpoint7=np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
    endpoint8=np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])
    endpoint9=np.array([[0, 0, 0], [0, 1, 1], [0, 0, 2]])
    
    br1=mah.morph.hitmiss(sk,branch1)
    br2=mah.morph.hitmiss(sk,branch2)
    br3=mah.morph.hitmiss(sk,branch3)
    br4=mah.morph.hitmiss(sk,branch4)
    br5=mah.morph.hitmiss(sk,branch5)
    br6=mah.morph.hitmiss(sk,branch6)
    br7=mah.morph.hitmiss(sk,branch7)
    br8=mah.morph.hitmiss(sk,branch8)
    br9=mah.morph.hitmiss(sk,branch9)
    
    ep1=mah.morph.hitmiss(sk,endpoint1)
    ep2=mah.morph.hitmiss(sk,endpoint2)
    ep3=mah.morph.hitmiss(sk,endpoint3)
    ep4=mah.morph.hitmiss(sk,endpoint4)
    ep5=mah.morph.hitmiss(sk,endpoint5)
    ep6=mah.morph.hitmiss(sk,endpoint6)
    ep7=mah.morph.hitmiss(sk,endpoint7)
    ep8=mah.morph.hitmiss(sk,endpoint8)
    ep9=mah.morph.hitmiss(sk,endpoint9)
    
    br = br1 + br2 + br3 + br4 + br5 + br6 + br7 + br8 + br9
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8 + ep9
    
    return br, ep



img, mask_download = download_img_mask(coord)
mask_download = mask_download[:1200,:]
img = img[:1200,:]

skeleton = morphology.medial_axis(mask_download == 255)

skeleton[0,:] = 0
skeleton[-1,:] = 0
skeleton[:,0] = 0
skeleton[:,-1] = 0

plt.figure()
plt.imshow(mask_download)
plt.figure()
plt.imshow(skeleton)
plt.figure()
plt.imshow(img)
plt.show()

skleton_sections = sort_skeleton(skeleton, img)
crops = []
H_list = []
W_list = []
mid_points = []
angles = []
print ">>>>>>>>>>>>>>>>>>>>>>>>"
one_section_segmentation = True
for section in skleton_sections:

    section_mid_points, section_angles, section_height, _angles = divide_skel(section, M=200)

    pdb.set_trace()
    plt.plot(_angles)
    plt.show()

    if len(section_mid_points) > 1:
        one_section_segmentation = False

    for index, _ in enumerate(section_mid_points):

        crop = extract_section(img, section_mid_points[index], section_angles[index], section_height[index])
        H, W = crop.shape[:2]
        print section_mid_points[index]
        print section_angles[index]
        print

        H_list.append(H)
        W_list.append(W)
        crops.append(crop)
        mid_points.append(section_mid_points[index])
        angles.append(section_angles[index])

#pdb.set_trace()

H_max = max(H_list)
W_max = max(W_list)

H_pad = H_max + int(ceil(float(H_max) / 32) * 32 - H_max)
W_pad = W_max + int(ceil(float(W_max) / 32) * 32 - W_max)

i_class = 0
with tf.Graph().as_default():

    image  = tf.placeholder(tf.float32, shape=[1, H_pad, W_pad, 3], name="input")
    logits = segnet_TF.segnet_extended(image)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    with tf.Session() as sess:

        sess.run(init_op)

        if one_section_segmentation:
            total_mask = np.zeros(skeleton.shape, dtype=np.uint8)
        else:
            total_mask = np.zeros(skeleton.shape).astype(bool)

        for crop, mid_point, angle in zip(crops, mid_points, angles):

            padded_crop, x_pad, y_pad = pad_img(crop, (H_pad, W_pad))
            padded_crop_ = padded_crop[...,::-1]
            padded_crop_ = padded_crop_[np.newaxis,:]
            padded_crop_ = padded_crop_.astype(np.float32)
            _logits = sess.run(logits, feed_dict={image: padded_crop_})
            mask = np.argmax(np.squeeze(_logits), axis=-1)
            mask = mask[y_pad[0]:H_pad-y_pad[-1], x_pad[0]:W_pad-x_pad[1]]

            if one_section_segmentation:
                plt.imshow(mask)
                plt.show()
                mask_lanes, num_lanes = segment_one_section(mask)

                for i_lane in range(1, num_lanes+1):

                    i_class += 1
                    total_mask_ = paste_mask(np.zeros(total_mask.shape).astype(bool), (mask_lanes == i_lane), mid_point, angle)
                    total_mask_ = binary_fill_holes(total_mask_.astype(int))
                    total_mask[total_mask_] = i_class

            else:
                total_mask = paste_mask(total_mask, (mask == 1), mid_point, angle)


        if one_section_segmentation:
            plt.imshow(total_mask)
            plt.show()
            vis_img = vis.vis_seg(img, total_mask, vis.make_palette(i_class+1))
            plt.imshow(vis_img)
            plt.show()
        else:
            total_mask = binary_fill_holes(total_mask.astype(int)).astype(int)
            total_mask = remove_small_objects(measure.label(total_mask, connectivity=2), min_size=50)
            new_total_mask, num_lanes = segment_marks(total_mask)
            vis_img = vis.vis_seg(img, new_total_mask, vis.make_palette(num_lanes+1))
            plt.imshow(vis_img)
            plt.show()



        



        

            
