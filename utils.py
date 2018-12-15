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

caffe_root = "/home/davidgj/projects_v2/caffe-segnet-cudnn5/"


sys.path.insert(0, caffe_root + 'python')

import caffe; caffe.set_mode_gpu()

prototxt = "/home/davidgj/projects_v2/SegNet-Tutorial/Models/roads/inference_video.prototxt"
caffemodel = "/home/davidgj/projects_v2/SegNet-Tutorial/Models/Inference/roads/snapshot_iter_1500/test_weights.caffemodel"
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

def road_segmentation(img):

    img = img[...,::-1]
    img = img.transpose((2,0,1))
    img = img[np.newaxis,...]
    print img.shape

    net.blobs['data'].reshape(*img.shape)
    net.blobs['data'].data[...] = img

    net.forward()

    predicted = np.squeeze(net.blobs['prob'].data)
    ind = np.argmax(predicted, axis=0)

    return ind

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
    
    #pad_amount = int(ceil(float(sz) / 32) * 32 - sz)
    pad_amount = sz_ - sz
    
    if pad_amount % 2:
        padding = (pad_amount / 2 , pad_amount - pad_amount / 2)
    else:
        padding = (pad_amount / 2, pad_amount / 2)
        
    return padding

def pad_img(img, pad_value=0):

    height, width = img.shape[:2]
    x_pad = get_padding(width, 224)
    y_pad = get_padding(height, 384)
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

def divide_skel(coords, D=50, M=200, thresh=5, min_height=100):
    
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
        if (angle - angles[ref_index]) > thresh:
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
    
    skel_labeled, num_labels = measure.label(skel, connectivity=2, return_num=True)
    sorted_coord_labels = []

    for i in range(num_labels):
        
        skel_cc = (skel_labeled == (i+1))
        
        if len(np.where(skel_cc)[0]) < min_num_points:
            continue
        
        _, ep = detect_br_ep(skel_cc)
        ep_y, ep_x = np.where(ep)
        ep_coords = zip(ep_y, ep_x)
        
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
            
        section_mid_points, section_angles, section_height, angles = divide_skel(sorted_coord)
        
        toltal_mask = np.zeros(img.shape).astype(bool)
        toltal_mask = toltal_mask[...,0]
        for index, _ in enumerate(section_mid_points):
            crop = extract_section(img, section_mid_points[index], section_angles[index], section_height[index])
            crop, pad_x, pad_y = pad_img(crop)
            mask = road_segmentation(crop)
            mask = mask[pad_y[0]:-pad_y[1], pad_x[0]:-pad_x[1]]
            mask = (mask == 1)
            toltal_mask = paste_mask(toltal_mask, mask, section_mid_points[index], section_angles[index])
            # pdb.set_trace()

        toltal_mask = binary_fill_holes(toltal_mask.astype(int)).astype(int)
        toltal_mask = remove_small_objects(measure.label(toltal_mask, connectivity=2), min_size=100)
        toltal_mask = (toltal_mask > 0).astype(np.uint8)
        props = measure.regionprops(measure.label(toltal_mask, connectivity=2), coordinates='xy')

        colineal_thresh = 15
        distance_thresh = 350
        colineal_points_total = []
        from PIL import Image, ImageDraw

        for prop_idx in range(len(props)):
            y0, x0 = props[prop_idx]["centroid"]
            orientation = props[prop_idx]["orientation"]
            line_coeffs = get_line_coeffs((x0, y0), orientation)
            # intersect_points = find_intesect_borders(line_coeffs, toltal_mask.shape)
            # im = Image.fromarray(toltal_mask)
            # draw = ImageDraw.Draw(im) 
            # draw.line([intersect_points[0], intersect_points[1]], fill=1, width=5)
            # plt.imshow(np.array(im))
            # plt.show()
            colineal_points = []
            distances = []
            for _prop_idx in range(len(props)):
                if _prop_idx != prop_idx:
                    _y0, _x0 = props[_prop_idx]["centroid"]
                    print get_dist_to_line(line_coeffs, (_x0, _y0))
                    print cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)]))
                    print _prop_idx

                    if get_dist_to_line(line_coeffs, (_x0, _y0)) < colineal_thresh:
                        distances.append(cdist(np.array([(x0,y0)]), np.array([(_x0,_y0)])))
                        if distances[-1] < distance_thresh:
                            colineal_points.append(_prop_idx)
                        else:
                            distances.pop()

            final_points = colineal_points         
            final_points = []

            colineal_points, distances = (list(t) for t in zip(*sorted(zip(colineal_points, distances), key = lambda x: x[1])))
            #pdb.set_trace()
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


        for prop_idx in range(len(props)):

            fig, ax = plt.subplots(1)
            ax.imshow(toltal_mask)

            y0, x0 = props[prop_idx]["centroid"]
            circle = Circle((x0, y0), 10, color='r')
            ax.add_patch(circle)

            for colineal_idx in range(len(colineal_points_total[prop_idx])):
                _prop_idx = colineal_points_total[prop_idx][colineal_idx]
                y0, x0 = props[_prop_idx]["centroid"]
                circle = Circle((x0, y0), 10, color='k')
                ax.add_patch(circle)

            plt.show()

            # circ1 = Circle(center[::-1],20)
            # circ2 = Circle(center[::-1],20)
            #pdb.set_trace()
            #colineal_points = [for ]

            #intersect_points = find_intesect_borders(line_coeffs, toltal_mask.shape)
        pdb.set_trace()

        # y0, x0 = props[0]["centroid"]
        # x1, y1 = x0 + 200 * math.cos(props[0]["orientation"] - math.pi/2), y0 + 200 * math.sin(props[0]["orientation"]-math.pi/2)

        from PIL import Image, ImageDraw
        im = Image.fromarray(toltal_mask)
        draw = ImageDraw.Draw(im) 
        draw.line([intersect_points[0], intersect_points[1]], fill=1, width=5)
        plt.imshow(np.array(im))



        pdb.set_trace()
        plt.figure()
        plt.imshow(vis.vis_seg(img, toltal_mask.astype(int), vis.make_palette(2)))
        plt.figure()
        plt.imshow(toltal_mask.astype(int))
        plt.show()
        sorted_coord_labels.append(sorted_coord)
        
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



img, mask = download_img_mask(coord)
mask = mask[:1200,:]
img = img[:1200,:]

skeleton = morphology.medial_axis(mask == 255)

skeleton[0,:] = 0
skeleton[-1,:] = 0
skeleton[:,0] = 0
skeleton[:,-1] = 0

plt.figure()
plt.imshow(skeleton)
plt.show()

skleton_points = sort_skeleton(skeleton, img)
        

            
