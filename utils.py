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
    

def _get_next_point(skel, current_point, prev_point=None):
    
    upper_limit = current_point[0]-1
    bottom_limit = current_point[0]+1
    left_limit = current_point[1]-1
    right_limit = current_point[1]+1
    neighbors = skel[upper_limit:bottom_limit+1, left_limit:right_limit+1]
    y, x = np.where(neighbors)
    neighbors_coord = zip(y,x)
    #neighbors_coord = [np.array(point) for point in zip(y,x)]
    #pdb.set_trace()
    if prev_point:
        relav_dist = _get_relav_dist(prev_point)
        prev_point = (1 - relav_dist[0], 1 - relav_dist[1])
    neighbors_coord = [point_coord for point_coord in neighbors_coord if point_coord not in [(1,1), prev_point]]
    #pdb.set_trace()
    next_point = neighbors_coord[np.argmin(cdist([(1,1)], neighbors_coord))]
    return next_point
    

def sort_skeleton(skel):
    
    skel_labeled, num_labels = measure.label(skel, connectivity=2, return_num=True)
    sorted_coord_labels = []

    for i in range(num_labels):
        #print "Entro"
        
        skel_cc = (skel_labeled == (i+1))
        
        _, ep = detect_br_ep(skel_cc)
        ep_y, ep_x = np.where(ep)
        ep_coords = zip(ep_y, ep_x)
        
        sorted_coord = []
        sorted_coord.append(ep_coords[0])
        #print sorted_coord[-1]
        _next_point = _get_next_point(skel_cc, sorted_coord[-1])
        relav_dist = _get_relav_dist(_next_point)
        next_point = (sorted_coord[-1][0] + relav_dist[0], sorted_coord[-1][1] + relav_dist[1])
        sorted_coord.append(next_point)
        #print sorted_coord[-1]
        
        while next_point != ep_coords[1]:
             
            current_point = next_point
            _next_point = _get_next_point(skel_cc, current_point, _next_point)
            relav_dist = _get_relav_dist(_next_point)
            next_point = (current_point[0] + relav_dist[0], current_point[1] + relav_dist[1])
            sorted_coord.append(next_point)
            #print sorted_coord[-1]
        
        new_skel_cc = np.zeros(skel_cc.shape,dtype=bool)
        new_skel_cc[zip(*sorted_coord)] = True
        pdb.set_trace()
        plt.figure()
        plt.imshow(new_skel_cc)
        plt.figure()
        plt.imshow(skel_cc)
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
    

#def skeleton_endpoints(skel):
    ## make out input nice, possibly necessary
    #skel = skel.copy()
    #skel[skel!=0] = 1
    #skel = np.uint8(skel)

    ## apply the convolution
    #kernel = np.uint8([[1,  1, 1],
                       #[1, 10, 1],
                       #[1,  1, 1]])
    #src_depth = -1
    #filtered = cv2.filter2D(skel,src_depth,kernel)

    ## now look through to find the value of 11
    ## this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    ##out = np.zeros_like(skel)
    ##out[np.where(filtered==11)] = 1
    #return np.where(filtered==11)

#def get_skeleton_graph(X):
    
    #G = nx.Graph()  # A graph to hold the nearest neighbours

    #tree = KDTree(X, leaf_size=2, metric='euclidean')  # Create a distance tree

    ## Now loop over your points and find the two nearest neighbours
    ## If the first and last points are also the start and end points of the line you can use X[1:-1]
    #for p in X:
        #dist, ind = tree.query(p, k=3)

        ## ind Indexes represent nodes on a graph
        ## Two nearest points are at indexes 1 and 2. 
        ## Use these to form edges on graph
        ## p is the current point in the list
        #G.add_node(p)
        #n1, l1 = X[ind[0][1]], dist[0][1]  # The next nearest point
        #n2, l2 = X[ind[0][2]], dist[0][2]  # The following nearest point  
        #G.add_edge(p, n1)
        #G.add_edge(p, n2)
        
    #return G

    

img, mask = download_img_mask(coord)
mask = mask[:1200,:]

#plt.figure()
#plt.imshow(img)

#plt.figure()
#plt.imshow(mask)


##pdb.set_trace()
#skeleton = morphology.skeletonize(mask == 255)

#plt.figure()
#plt.imshow(skeleton)


skeleton = morphology.medial_axis(mask == 255)

skeleton[0,:] = 0
skeleton[-1,:] = 0
skeleton[:,0] = 0
skeleton[:,-1] = 0

skleton_points = sort_skeleton(skeleton)
print skleton_points[0][:10]
plt.figure()
plt.imshow(skeleton)
plt.show()



#fig,ax = plt.subplots(1)
#ax.imshow(skeleton)

##y, x = skeleton_endpoints(skeleton)

#br, ep = detect_br_ep(skeleton)

#y, x = np.where(ep)

#for xx,yy in zip(x,y):
    #circ = Circle((xx,yy),20)
    #ax.add_patch(circ)

##skeleton_list = np.where(skeleton > 0)
##pdb.set_trace()
##get_skeleton_graph(skeleton_list)

## Display the image and plot all contours found
#fig, ax = plt.subplots()
#ax.imshow(skeleton, interpolation='nearest', cmap=plt.cm.gray)
#contours = measure.find_contours(skeleton, 0)
#for n, contour in enumerate(contours):
    #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

#ax.axis('image')
#ax.set_xticks([])
#ax.set_yticks([])
#plt.show()

#plt.show()
        

            
