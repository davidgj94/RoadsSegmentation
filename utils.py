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
from sklearn.neighbors import KDTree
import networkx as nx
import cv2
from matplotlib.patches import Circle

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

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    #out = np.zeros_like(skel)
    #out[np.where(filtered==11)] = 1
    return np.where(filtered==11)

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

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(mask)


#pdb.set_trace()
skeleton = morphology.skeletonize(mask == 255)

plt.figure()
plt.imshow(skeleton)


skeleton = morphology.medial_axis(mask == 255)

plt.figure()
plt.imshow(skeleton)


skeleton[0,:] = 0
skeleton[-1,:] = 0
skeleton[:,0] = 0
skeleton[:,-1] = 0

fig,ax = plt.subplots(1)
ax.imshow(skeleton)

y, x = skeleton_endpoints(skeleton)

for xx,yy in zip(x,y):
    circ = Circle((xx,yy),7)
    ax.add_patch(circ)

#skeleton_list = np.where(skeleton > 0)
#pdb.set_trace()
#get_skeleton_graph(skeleton_list)

plt.show()
        

            
