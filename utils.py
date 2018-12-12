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

def get_padding(sz):
    
    pad_amount = int(ceil(float(sz) / 32) * 32 - sz)
    
    if pad_amount % 2:
        
        padding = (pad_amount / 2 , pad_amount - pad_amount / 2)
    else:
        padding = (pad_amount / 2, pad_amount / 2)
        
return padding

def pad_img(img, pad_value=0):

    height, width = img.shape[:2]
    x_pad = get_padding(width)
    y_pad = get_padding(height)
    img_padded = cv2.copyMakeBorder(down_img, y_pad[0], y_pad[1], x_pad[0], x_pad[1], cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    return img_padded



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
    circ1 = Circle(center[::-1],20)
    circ2 = Circle(center[::-1],20)
    circ3 = Circle(center[::-1],20)
    fig1,ax1 = plt.subplots(1)
    ax1.imshow(img)
    ax1.add_patch(circ1)
    fig2,ax2 = plt.subplots(1)
    ax2.imshow(rotated)
    ax2.add_patch(circ2)
    crop = cv2.getRectSubPix(rotated, (width, height), center[::-1])
    fig1,ax3 = plt.subplots(1)
    ax3.imshow(crop)
    
    toltal_mask = paste_mask(np.zeros(img.shape).astype(bool), crop, center, angle)
    ax3.imshow(toltal_mask)
    ax3.add_patch(circ3)
    plt.show()
    
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
        extract_section(skel_cc.astype(np.uint8), section_mid_points[0], section_angles[0], section_height[0])
        
        for index, _ in enumerate(section_mid_points):
            extract_section(img, section_mid_points[index], section_angles[index], section_height[index])
        
        divide_skel(coords)
        extract_section(img, center, angle, height, width=200)
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
        

            
