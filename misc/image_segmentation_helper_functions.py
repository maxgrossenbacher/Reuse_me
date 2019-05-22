import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
import sys
import glob
import datetime
import pickle
import time

from rsgislib.segmentation import segutils
import rsgislib
from rsgislib import imagefilter
from rsgislib import imageutils
from rsgislib import rastergis
from rsgislib.rastergis import ratutils
import gdal
import rios

from skimage import graph, data, io, segmentation, color
from skimage.segmentation import slic
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.future import graph
from skimage.measure import regionprops
from skimage.color import rgb2lab, rgb2xyz, xyz2lab, rgb2hsv
from skimage import exposure
from skimage import draw

import traceback
import cv2
import heapq
import imutils
# from imutils import build_montages
from imutils import paths

import multiprocessing as mp
from collections import defaultdict

from libtiff import TIFF

# module_path = os.path.abspath(os.path.join('helper_functions'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import validation
import functions as fct
import extraction_helper as eh
warnings.filterwarnings('ignore')
plt.style.use('classic')

###############################################################################
# Get Image from Azure and output np.array()
def visualize(path, title=None, plot=False):
    '''
    DESC: Visualize image from img_path and output np.array
    INPUT: img_path=str, optional(title=str), plot(bool)
    -----
    OUTPUT: np.array with dtype uint8
    '''
    img = TIFF.open(path, mode="r")
    img = img.read_image()
    img = eh.rgb_standardization(eh.minmax_scaling(eh.normalization(img, 65535.)))
    img = img.astype(np.uint8)
#     print (np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]), np.mean(img[:,:,3]))
    plt.imshow(img[:,:,:3])
    plt.title(title)
    if plot:
        plt.show()
    return img

###############################################################################
# Griffin Features - processing bands and generating image features
def process_bands(original_bands, source='PlanetScope'):
    '''
    DESC: Processes DOVE/PlaentScope and RapidEye images by band
    INPUT: original_bands = np.array of image with shape (x,y,num_bands)
    '''
    reference_image = original_bands[:,:,2].copy() / 1.0
    reference_image[reference_image == 0.0] = np.nan
    image_red = original_bands[:,:,2].copy() / 65535.0
    image_red[np.isnan(image_red)] = 0

    image_green = original_bands[:,:,1].copy() / 65535.0
    image_green[np.isnan(image_green)] = 0

    image_blue = original_bands[:,:,0].copy() / 65535.0
    image_blue[np.isnan(image_blue)] = 0

    image_nir = original_bands[:,:,3].copy() / 65535.0
    image_nir[np.isnan(image_nir)] = 0

    if source == 'RapidEye':
            image_rededge = original_bands['rededge'].copy() / 65535.0
            image_rededge[np.isnan(image_rededge)] = 0

    image_bgr = np.dstack((image_red, image_green, image_blue))
    ret_bgr = image_bgr.copy()
    # ret_bgr = exposure.adjust_log(ret_bgr)
    ret_bgr = exposure.equalize_adapthist(ret_bgr, clip_limit=0.04)

    hsv = rgb2hsv(image_bgr)
    image_h = hsv[:, :, 0]
    image_s = hsv[:, :, 1]
    image_v = hsv[:, :, 2]

    lab = rgb2lab(image_bgr)
    image_l = lab[:, :, 0]
    image_a = lab[:, :, 1]
    image_b = lab[:, :, 2]

    xyz = rgb2xyz(image_bgr)
    image_x = xyz[:, :, 0]
    image_y = xyz[:, :, 1]
    image_z = xyz[:, :, 2]

    clab = xyz2lab(xyz)
    image_cl = clab[:, :, 0]
    image_ca = clab[:, :, 1]
    image_cb = clab[:, :, 2]
    return ret_bgr, reference_image, image_red, image_green, image_blue, image_nir, \
           image_h, image_s, image_v, image_l, image_a, image_b, image_x, image_y, image_z, \
           image_cl, image_ca, image_cb

################################################################################
# Helper functions for creating Griffin Features

def getMean(indx, image, reference_image):
    referenceData = reference_image[indx[0], indx[1]]
    data = image[indx[0], indx[1]]
    data = data[~np.isnan(referenceData)]
    if len(data) == 0:
        return np.nan
    else:
        p1 = np.percentile(data, 25)
        p2 = np.percentile(data, 75)
        data = data[data > p1]
        data = data[data < p2]
        return np.mean(data)

def getImageMean(image, reference_image):
    image = image[~np.isnan(reference_image)]
    return np.mean(image)

def get_SR(image_nir, image_red, indx, reference_image):
    referenceData = reference_image[indx[0], indx[1]]
    data_nir = image_nir[indx[0], indx[1]]
    data_nir = data_nir[~np.isnan(referenceData)]
    # data_nir = data_nir[np.nonzero(data_nir)]
    data_red = image_red[indx[0], indx[1]]
    data_red = data_red[~np.isnan(referenceData)]
    # data_red = data_red[np.nonzero(data_red)]
    return np.mean(np.divide(data_nir, data_red + 1)), data_nir, data_red

def get_EVI(data_nir, data_red, image_blue, indx, reference_image):
    referenceData = reference_image[indx[0], indx[1]]
    data_blue = image_blue[indx[0], indx[1]]
    data_blue = data_blue[~np.isnan(referenceData)]
    # data_blue = data_blue[np.nonzero(data_blue)]
    return np.mean(2.5 * np.divide((data_nir - data_red), (1.0 + data_nir + (6.0 * data_red) - (7.5 * data_blue)) + 1.0)), data_blue

def get_CL_green(data_nir, image_green, indx, reference_image):
    referenceData = reference_image[indx[0], indx[1]]
    data_green = image_green[indx[0], indx[1]]
    data_green = data_green[~np.isnan(referenceData)]
    # data_green = data_green[np.nonzero(data_green)]
    return np.mean(np.divide(data_nir, data_green + 1.0) - 1.0), data_green

def get_MTCI(data_nir, data_rededge, data_red):
    return np.mean(np.divide((data_nir - data_rededge),(data_rededge - data_red) + 1.0))

def get_data_blue(image_blue, indx, reference_image):
    referenceData = reference_image[indx[0], indx[1]]
    data_blue = image_blue[indx[0], indx[1]]
    data_blue = data_blue[~np.isnan(referenceData)]
    return data_blue

################################################################################
# Griffin Features

def extractFeatures(reference_image, cluster_segments, cluster_list, image_red, image_green, image_blue, image_nir, image_h, image_s, image_v,
                    image_l, image_a, image_b, image_x, image_y, image_z, image_cl, image_ca, image_cb, image_rededge, img_date, source):
    '''
    DESC: Griffin Feature generation - create a df of features by band per segement for image
    INPUT: reference_image = np.array, cluster_segments=np.array segment mask, cluster_list=list of unique segments,
            image_red - image_rededge = output from process_bands fxn,
            img_date=datetime object [year,month,day] (tuple), source=str()- PlaentScope or RapidEye
    '''

    if source == 'RapidEye':
        day_of_year = float(img_date.timetuple().tm_yday) / 365.0
    else:
        day_of_year = float(img_date.timetuple().tm_yday) / 365.0

    image_mean_red = getImageMean(image_red, reference_image)
    image_mean_green = getImageMean(image_green, reference_image)
    image_mean_blue = getImageMean(image_blue, reference_image)
    image_mean_rededge = 0
    if source == 'RapidEye':
        image_mean_rededge = getImageMean(image_rededge, reference_image)
    image_mean_nir = getImageMean(image_nir, reference_image)

    image_mean_h = getImageMean(image_h, reference_image)
    image_mean_s = getImageMean(image_s, reference_image)
    image_mean_v = getImageMean(image_v, reference_image)
    image_mean_l = getImageMean(image_l, reference_image)
    image_mean_a = getImageMean(image_a, reference_image)
    image_mean_b = getImageMean(image_b, reference_image)
    image_mean_x = getImageMean(image_x, reference_image)
    image_mean_y = getImageMean(image_y, reference_image)
    image_mean_z = getImageMean(image_z, reference_image)
    image_mean_cl = getImageMean(image_cl, reference_image)
    image_mean_ca = getImageMean(image_ca, reference_image)
    image_mean_cb = getImageMean(image_cb, reference_image)

    features = dict()
    features['day_of_year'] = []
    features['SR'] = []
    features['CL_green'] = []
    if source == 'RapidEye':
        features['CL_rededge'] = []
        features['MTCI'] = []

    features['red_mean'] = []
    features['green_mean'] = []
    features['blue_mean'] = []
    if source == 'RapidEye':
        features['rededge_mean'] = []

    features['nir_mean'] = []
    features['segment']=[]

    features['h_mean'] = []
    features['s_mean'] = []
    features['v_mean'] = []
    features['l_mean'] = []
    features['a_mean'] = []
    features['b_mean'] = []
    features['x_mean'] = []
    features['y_mean'] = []
    features['z_mean'] = []
    features['cl_mean'] = []
    features['ca_mean'] = []
    features['cb_mean'] = []

    features['image_mean_red'] = []
    features['image_mean_green'] = []
    features['image_mean_blue'] = []
    if source == 'RapidEye':
        features['image_mean_rededge'] = []
    features['image_mean_nir'] = []

    features['image_mean_h'] = []
    features['image_mean_s'] = []
    features['image_mean_v'] = []
    features['image_mean_l'] = []
    features['image_mean_a'] = []
    features['image_mean_b'] = []
    features['image_mean_x'] = []
    features['image_mean_y'] = []
    features['image_mean_z'] = []
    features['image_mean_cl'] = []
    features['image_mean_ca'] = []
    features['image_mean_cb'] = []

    features['normalized_R'] = []
    features['normalized_G'] = []
    features['normalized_B'] = []

    features['mean_R_by_B'] = []
    features['mean_R_by_B_plus_R'] = []
    features['mean_chroma'] = []

    features['R-G'] = []
    features['R-B'] = []
    features['G-R'] = []
    features['G-B'] = []
    features['B-R'] = []
    features['B-G'] = []

    for cluster_num in cluster_list:
        cluster_indx = np.where(cluster_segments == cluster_num)
        features['day_of_year'].append(day_of_year)

        sr, data_nir, data_red = get_SR(image_nir, image_red, cluster_indx, reference_image)
        features['SR'].append(sr)

        data_blue = get_data_blue(image_blue, cluster_indx, reference_image)
        cl_green, data_green = get_CL_green(data_nir, image_green, cluster_indx, reference_image)
        features['CL_green'].append(cl_green)

        if source == 'RapidEye':
            cl_rededge, data_rededge = get_CL_green(data_nir, image_rededge, cluster_indx, reference_image)
            features['CL_rededge'].append(cl_rededge)
            features['MTCI'].append(get_MTCI(data_nir, data_rededge, data_red))

        features['red_mean'].append(getMean(cluster_indx, image_red, reference_image))
        features['green_mean'].append(getMean(cluster_indx, image_green, reference_image))
        features['blue_mean'].append(getMean(cluster_indx, image_blue, reference_image))
        if source == 'RapidEye':
            features['rededge_mean'].append(getMean(cluster_indx, image_rededge, reference_image))
        features['nir_mean'].append(getMean(cluster_indx, image_nir, reference_image))

        features['segment'].append(cluster_num)

        features['h_mean'].append(getMean(cluster_indx, image_h, reference_image))
        features['s_mean'].append(getMean(cluster_indx, image_s, reference_image))
        features['v_mean'].append(getMean(cluster_indx, image_v, reference_image))
        features['l_mean'].append(getMean(cluster_indx, image_l, reference_image))
        features['a_mean'].append(getMean(cluster_indx, image_a, reference_image))
        features['b_mean'].append(getMean(cluster_indx, image_b, reference_image))
        features['x_mean'].append(getMean(cluster_indx, image_x, reference_image))
        features['y_mean'].append(getMean(cluster_indx, image_y, reference_image))
        features['z_mean'].append(getMean(cluster_indx, image_z, reference_image))
        features['cl_mean'].append(getMean(cluster_indx, image_cl, reference_image))
        features['ca_mean'].append(getMean(cluster_indx, image_ca, reference_image))
        features['cb_mean'].append(getMean(cluster_indx, image_cb, reference_image))

        features['image_mean_red'].append(image_mean_red)
        features['image_mean_green'].append(image_mean_green)
        features['image_mean_blue'].append(image_mean_blue)
        if source == 'RapidEye':
            features['image_mean_rededge'].append(image_mean_rededge)
        features['image_mean_nir'].append(image_mean_nir)

        features['image_mean_h'].append(image_mean_h)
        features['image_mean_s'].append(image_mean_s)
        features['image_mean_v'].append(image_mean_v)
        features['image_mean_l'].append(image_mean_l)
        features['image_mean_a'].append(image_mean_a)
        features['image_mean_b'].append(image_mean_b)
        features['image_mean_x'].append(image_mean_x)
        features['image_mean_y'].append(image_mean_y)
        features['image_mean_z'].append(image_mean_z)
        features['image_mean_cl'].append(image_mean_cl)
        features['image_mean_ca'].append(image_mean_ca)
        features['image_mean_cb'].append(image_mean_cb)

        features['normalized_R'].append(np.mean(np.divide(data_red, (data_red + data_green + data_blue + 1.0))))
        features['normalized_G'].append(np.mean(np.divide(data_green, (data_red + data_green + data_blue + 1.0))))
        features['normalized_B'].append(np.mean(np.divide(data_blue, (data_red + data_green + data_blue + 1.0))))

        features['mean_R_by_B'].append(np.mean(np.divide(data_red, data_blue + 1.0)))
        features['mean_R_by_B_plus_R'].append(np.mean(np.divide(data_red, data_blue + data_red + 1.0)))
        try:
            features['mean_chroma'].append(max(np.nanmax(data_red), np.nanmax(data_green), np.nanmax(data_blue)) - \
                              min(np.nanmin(data_red), np.nanmin(data_green), np.nanmin(data_blue)))
        except ValueError:
            features['mean_chroma'].append(np.nan)

        features['R-G'].append(np.mean(data_red - data_green))
        features['R-B'].append(np.mean(data_red - data_blue))
        features['G-R'].append(np.mean(data_green - data_red))
        features['G-B'].append(np.mean(data_green - data_blue))
        features['B-R'].append(np.mean(data_blue - data_red))
        features['B-G'].append(np.mean(data_blue - data_green))

    df = pd.DataFrame.from_dict(features)
    return df

###############################################################################
# Utlitiy Functions

def load_obj(filepath):
    '''
    DESC: Load object as pickle from filepath
    INPUT: filepath = str()
    -----
    OUTPUT: loads pickled objected
    '''
    import dill
    with open(filepath ,'rb') as f:
        return dill.load(f)

def save_obj(obj, filepath):
    '''
    DESC: Save object as pickle to filepath
    INPUT: obj=(list, dict, etc.), filepath = str()
    -----
    OUTPUT: saves pickled object to filepath
    '''
    import dill
    with open(filepath ,'wb') as f:
        return dill.dump(obj, f, protocol=2)

def sp_idx(s):
    '''
    DESC: creates a flattened array/list of segments with pixel values
    INPUT: segment np.array
    -----
    OUTPUT: list of segments with pixel values
    '''
    u = np.unique(s)
    return [np.where(s == i) for i in u]

def rgb_metric(s, metric='sd'):
    '''
    DESC: calcuates (R-G)/B per pixel
    INPUT: segment_list from (sp_idx fxn), metric=str() ['sd', 'mean']
    -----
    OUTPUT: np.nanstd or np.nanmean for each segment/superpixel
    '''
    B=s[:,0]
    G=s[:,1]
    R=s[:,2]
    metric = (R-G)/B
    if 'sd':
        return np.nanstd(metric)
    if 'mean':
        return np.nanmean(metric)

def grayscale_metric(s, metric='sd'):
    '''
    DESC: calcuates mean/sd of grayscale (R+G+B)/3 per segment/superpixel
    INPUT: segment_list from (sp_idx fxn), metric=str() ['sd', 'mean']
    -----
    OUTPUT: np.nanstd or np.nanmean for each segment/superpixel
    '''
    B=s[:,0]
    G=s[:,1]
    R=s[:,2]
    metric = (R+G+B)/3
    if 'sd':
        return np.nanstd(metric)
    if 'mean':
        return np.nanmean(metric)

################################################################################
# Merging segments by mean color - http://scikit-image.org/docs/dev/api/skimage.future.graph.html#skimage.future.graph.merge_hierarchical

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.
    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.
    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    # networkx updates data dictionary if edge exists
    # this would mean we have to reposition these edges in
    # heap if their weight is updated.
    # instead we invalidate them

    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            # invalidate edges incident on `dst`, they have new weights
            data['heap item'][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            # will handle the case where the edge did not exist in the existing
            # graph
            pass

        wt = data['weight']
        heap_item = [wt, node, nbr, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)


def _rename_node(graph, node_id, copy_id):
    """ Rename `node_id` in `graph` to `copy_id`. """

    graph._add_node_silent(copy_id)
    graph.node[copy_id].update(graph.node[node_id])

    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]['weight']
        graph.add_edge(nbr, copy_id, {'weight': wt})

    graph.remove_node(node_id)


def _invalidate_edge(graph, n1, n2):
    """ Invalidates the edge (n1, n2) in the heap. """
    graph[n1][n2]['heap item'][3] = False


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

def merge_hierarchical_segments(labels, rag, segments, rag_copy, in_place_merge,
                               merge_func, weight_func):
    """Perform hierarchical merging of a RAG.
    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.
    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.
    Returns
    -------
    out : ndarray
        The new labeled array.
    """
    if rag_copy:
        rag = rag.copy()

    edge_heap = []
    for n1, n2, data in rag.edges(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    while len(edge_heap) > 0 and len(rag.nodes()) > segments:
        _, n1, n2, valid = heapq.heappop(edge_heap)

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted

            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            merge_func(rag, src, dst)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix

    return label_map[labels]


###############################################################################
# Not really necessary function

def try_different_num_segments(image, num_segments=(5, 10, 15, 20)):
    # loop over the number of segments
    segments=[]
    for num in num_segments:
#         print("Superpixels -- {} segments" .format(num))
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        seg = slic(img_as_float(image),
                        n_segments = num,
                        sigma=5,
                        max_iter=100,
                        compactness=10,
                   enforce_connectivity=True,
                       slic_zero=True)

        # show the output of SLIC
        fig = plt.figure("Superpixels")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), seg))
        plt.show()
        seg = seg + 1
        segments.append(seg)
    return segments

def segment_image(image, n_segments, standardize=None):
    '''
    DESC: Segment image
    INPUT: images=np.array(), n_segemnts=int(), standardize=int()
    -----
    OUTPUT: returns segment np.array and number of original segments created
    '''
    # loop over the number of segments
#     print("Superpixels -- {} segments" .format(n_segments))
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    seg = slic(img_as_float(image),
                    n_segments = n_segments,
                    sigma=5,
                    max_iter=200,
                   slic_zero=True)
    original_segs = len(np.unique(seg))

    if len(np.unique(seg)) > standardize:
        g = graph.rag_mean_color(image, seg)
        seg = merge_hierarchical_segments(seg, g, segments=standardize, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)
    seg = seg + 1
    return seg, original_segs

def image_preparation(df, img_path):
    # get img_path and date and fieldID and filename
    ind = np.where(df['img_path']==img_path)
    f = df['filename'].iloc[ind].to_string(index=False, header=True)

    f = f.split(' ')[-1]

    date = df['date'].iloc[ind].to_string(index=False, header=True)
    year = int(date[:4])
    month = int(date[4:6])
    day =int(date[6:])
    img_date = datetime.datetime(year,month,day)

    # original image
    image_raw = visualize(img_path, plot=False)
    return image_raw, img_date, f


def get_griffin_features(image_raw, segment, img_date, filename):
    # Process image
    ret_bgr, reference_image, image_red, image_green, image_blue, image_nir, \
    image_h, image_s, image_v, image_l, image_a, image_b, image_x, image_y, image_z, \
    image_cl, image_ca, image_cb = process_bands(image_raw)
    griffin_df= extractFeatures(
                            reference_image = reference_image,
                            cluster_segments=segment,
                            cluster_list=np.unique(segment),
                            image_red=image_red,
                            image_green=image_green,
                            image_blue=image_blue,
                            image_nir=image_nir,
                            image_h=image_h,
                            image_s=image_s,
                            image_v=image_v,
                            image_l=image_l,
                            image_a=image_a,
                            image_b=image_b,
                            image_x=image_x,
                            image_y=image_y,
                            image_z=image_z,
                            image_cl=image_cl,
                            image_ca=image_ca,
                            image_cb=image_cb,
                            image_rededge=None,
                            img_date=img_date,
                            source='PlanetScope')
    griffin_df['filename'] = filename
    return griffin_df


def SLIC_segmentation(image_raw, n_segments=10, standardize=8):
    image = image_raw[:,:,:3]
    image = img_as_float(image)
    image = image.astype(np.float32)
    # Super Pixel Segmentaiton
    segment, original_segs = segment_image(image, n_segments=n_segments, standardize=standardize)
    return segment, original_segs


def get_SLIC_segmentation(df, fieldIDs, files_list=[],n_segments=10, standardize=8):
    images, segments, original_segs ={},{},{}
    # rgbs, grays = {},{}
    dfs = []
    if isinstance(fieldIDs,int):
        fieldIDs = [fieldIDs]
    selected_field = df[df['fieldID'].isin(fieldIDs)]
    if len(files_list) > 0:
        selected_field = selected_field[selected_field['filename'].isin(files_list)]
    for x in selected_field['img_path']:
        # process image
        image_raw, img_date, filename = image_preparation(selected_field, x)
        # segmentation using SLIC
        segment, original_seg = SLIC_segmentation(image_raw[:,:,:3], n_segments=n_segments, standardize=standardize)

        # superpixel_list = sp_idx(segment)
        # superpixel = [image_raw[:,:,:3][idx] for idx in superpixel_list]
        # rgb_std_segment = [rgb_metric(s, metric='sd') for s in superpixel]
        # gray_std_segment = [grayscale_metric(s, metric='sd') for s in superpixel]

        # Get Griffin Features
        griffin_df = get_griffin_features(image_raw, segment, img_date, filename)

        # Collect data
        dfs.append(griffin_df)
        images[filename] = image_raw
        segments[filename] = segment
        original_segs[filename] = original_seg
        # rgbs[filename] = rgb_std_segment
        # grays[filename] = gray_std_segment
    dfs_field = pd.concat(dfs, axis=0)
    return images, segments, original_segs, dfs_field


# def get_image_segmentation(df, fieldIDs, files_list=[],n_segments=10, standardize=8, save_dir=None, number_fields=np.inf):
#     months = {0:'jan',1:'feb',2:'mar',3:'apr',4:'may',5:'june', 6:'july', 7:'aug', 8:'sept', 9:'oct', 10:'nov', 11:'dec'}
#     images, segments, original_segs ={},{},{}
#     dfs = []
#     if isinstance(fieldIDs,int):
#         fieldIDs = [fieldIDs]
#     selected_field = df[df['fieldID'].isin(fieldIDs)]
#     if len(files_list) > 0:
#         selected_field = selected_field[selected_field['filename'].isin(files_list)]
#     count = 0
#     for x in selected_field['img_path']:
#         if count < number_fields:
#             count += 1
#             # get img_path and date and fieldID and filename
#             ind = np.where(selected_field['img_path']==x)
#             f = selected_field['filename'].iloc[ind].to_string(index=False, header=True)
#             i = selected_field['fieldID'].iloc[ind].to_string(index=False, header=True)
#             f = f.split(' ')[-1]
#
#             date = selected_field['date'].iloc[ind].to_string(index=False, header=True)
#             year = int(date[:4])
#             month = int(date[4:6])
#             day =int(date[6:])
#             img_date = datetime.datetime(year,month,day)
#
#             # original image
#             image_raw = visualize(x)
#
#             # Process image
#             ret_bgr, reference_image, image_red, image_green, image_blue, image_nir, \
#             image_h, image_s, image_v, image_l, image_a, image_b, image_x, image_y, image_z, \
#             image_cl, image_ca, image_cb = process_bands(image_raw)
#
#             image = image_raw[:,:,:3]
#             image = img_as_float(image)
#             image = image.astype(np.float32)
#
#             # Super Pixel Segmentaiton
#             segment, original_seg = segment_image(image, n_segments=n_segments, standardize=standardize)
#
# #             superpixel_list = sp_idx(segment)
# #             superpixel = [image_raw[:,:,:3][idx] for idx in superpixel_list]
# #             rgb_std_segment = [rgb_metric(s, metric='sd') for s in superpixel]
# #             gray_std_segment = [grayscale_metric(s, metric='sd') for s in superpixel]
#
#             griffin_df= extractFeatures(
#                                         reference_image = reference_image,
#                                         cluster_segments=segment,
#                                         cluster_list=np.unique(segment),
#                                         image_red=image_red,
#                                         image_green=image_green,
#                                         image_blue=image_blue,
#                                         image_nir=image_nir,
#                                         image_h=image_h,
#                                         image_s=image_s,
#                                         image_v=image_v,
#                                         image_l=image_l,
#                                         image_a=image_a,
#                                         image_b=image_b,
#                                         image_x=image_x,
#                                         image_y=image_y,
#                                         image_z=image_z,
#                                         image_cl=image_cl,
#                                         image_ca=image_ca,
#                                         image_cb=image_cb,
#                                         image_rededge=None,
#                                         img_date=img_date,
#                                         source='PlanetScope')
#             griffin_df['filename'] = f
#             dfs.append(griffin_df)
#
#             original_segs[f] = original_seg
#
#             images[f] = image_raw
#             segments[f] = segment
#     dfs_field = pd.concat(dfs, axis=0)
#     return images,segments,original_segs, dfs_field

################################################################################
# RSGISLib segmentation

def RSGISLib_segmentation(img_path, save_name, save_dir, numClusters=10, minPxls=5000, distThres=500):
    save_dir = save_dir+'/'+img_path.split("/")[-1]+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_clump = save_dir+'{}_clumps.kea'.format(save_name)
    mean = save_dir+'{}_mean.kea'.format(save_name)
    json =save_dir+'{}_json'.format(save_name)
    imgstretchstats = save_dir+'{}_imgstretchstats.txt'.format(save_name)
    kmeans = save_dir+'{}_kmeans'.format(save_name)
    segutils.runShepherdSegmentation(inputImg=img_path,
                                     outputClumps=output_clump,
                                     outputMeanImg=mean,
                                     minPxls=minPxls,
                                     numClusters=numClusters,
                                    saveProcessStats=True,
                                     distThres=distThres,
                                     imgStretchStats = imgstretchstats,
                                     kMeansCentres = kmeans,
                                    imgStatsJSONFile=json)

    outascii = save_dir+'{}_imgstats.txt'.format(save_name)
    ratutils.populateImageStats(img_path, output_clump, outascii=outascii,
                                threshold=0.0, calcMin=True, calcMax=True,
                                calcSum=True, calcMean=True, calcStDev=True,
                                calcMedian=False, calcCount=False, calcArea=False,
                                calcLength=False, calcWidth=False, calcLengthWidth=False)

    imageutils.popImageStats(output_clump, True, 0, True)


    outimage = save_dir+"{}_test_gdal.kea".format(save_name)
    gdalformat = 'KEA'
    datatype = rsgislib.TYPE_32FLOAT
    fields = ['Histogram', 'Red', 'Green', 'Blue', 'Alpha']
    rastergis.exportCols2GDALImage(output_clump, outimage, gdalformat, datatype, fields)

    output = save_dir+'{}_array.txt'.format(save_name)
    rastergis.export2Ascii(output_clump, outfile=output, fields=fields)


    ds = gdal.Open(outimage)
    myarray = np.array(ds.ReadAsArray())
    new = np.dstack((myarray[1,:,:], myarray[2,:,:], myarray[3,:,:]))
    new = new.astype(np.uint8)
    segment = (new[:,:,0]+new[:,:,1]+new[:,:,2])/3

    return segment, new, myarray

def get_RSGISLib_segmentation(df, fieldIDs, save_dir,files_list=[],standardize=8, numClusters=10, minPxls=5000, distThres=500):
    images, segments, seg_dicts, original_segs ={},{},{},{}
    # rgbs, grays ={},{}
    dfs = []
    if isinstance(fieldIDs,int):
        fieldIDs = [fieldIDs]
    selected_field = df[df['fieldID'].isin(fieldIDs)]
    if len(files_list) > 0:
        selected_field = selected_field[selected_field['filename'].isin(files_list)]
    for img_path in selected_field['img_path']:
        seg_dict = {}
        # process image
        try:
            image_raw, img_date, filename = image_preparation(selected_field, img_path)
            # segmentation using RSGISLib
            segment, new, myarray = RSGISLib_segmentation(img_path,
                                                          save_name=filename,
                                                          save_dir=save_dir,
                                                          numClusters=numClusters,
                                                          minPxls=minPxls,
                                                          distThres=distThres)
            for ind, s in enumerate(np.unique(segment)):
                segment[segment==s] = int(ind+1)
            segment = segment.astype(np.uint8)
            original_seg = np.unique(segment)

            # standardize
            if len(np.unique(segment)) > standardize:
                g = graph.rag_mean_color(image_raw[:,:,:3], segment)
                segment = merge_hierarchical_segments(segment, g, segments=standardize, rag_copy=False,
                                                   in_place_merge=True,
                                                   merge_func=merge_mean_color,
                                                   weight_func=_weight_mean_color)

            # relabel segments by brightness
            new_segment = segment.copy()
            for ind, s in enumerate(sorted(np.unique(segment))):
                seg_dict[s] = ind+1
            for k, v in seg_dict.items():
                new_segment[segment==k] = v
            new_segment = new_segment.astype(np.uint8)

            # superpixel_list = sp_idx(new_segment)
            # superpixel = [image_raw[:,:,:3][idx] for idx in superpixel_list]
            # rgb_std_segment = [rgb_metric(s, metric='sd') for s in superpixel]
            # gray_std_segment = [grayscale_metric(s, metric='sd') for s in superpixel]


            # Get Griffin Features
            griffin_df = get_griffin_features(image_raw, new_segment, img_date, filename)

            # Collect data
            dfs.append(griffin_df)
            seg_dicts[filename] = seg_dict
            segments[filename] = new_segment
            original_segs[filename] = original_seg
            images[filename] = image_raw
            # rgbs[filename] = rgb_std_segment
            # grays[filename] = gray_std_segment
        except:
            pass
    dfs_field = pd.concat(dfs, axis=0)
    return images, segments,original_segs, dfs_field, seg_dicts

###############################################################################
# Image Colorfulness equations - https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/

def image_colorfulness(image):
    '''
    DESC: Get Colorfulness value of an image
    INPUT: image=np.array()
    -----
    OUTPUT: Colorfulness value
    '''
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def segment_colorfulness(image, mask):
    '''
    DESC: Get Colorfulness value of an image segment
    INPUT: image=np.array(), mask=segment mask np.array()
    -----
    OUTPUT: Colorfulness value
    '''
    # split the image into its respective RGB components, then mask
    # each of the individual RGB channels so we can compute
    # statistics only for the masked region
    (B, G, R) = cv2.split(image.astype("float"))
    R = np.ma.masked_array(R, mask=mask)
    G = np.ma.masked_array(B, mask=mask)
    B = np.ma.masked_array(B, mask=mask)

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`,
    # then combine them
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

################################################################################
# Image Segmentation features

def NDVI_r(img):
    if len(img.shape) == 3:
        return 1.0 * ((img[:, :, 3]-img[:, :, 2]) / (img[:, :, 3] + img[:, :, 2]))
    elif len(img.shape) == 4:
        return 1.0 * ((img[:, :, 3, :]-img[:, :, 2, :]) / (img[:, :, 3, :] + img[:, :, 2, :]))
    elif len(img.shape) == 2:
        return 1.0 * ((img[:, 3]-img[:, 2]) / (img[:, 3] + img[:, 2]))

def NDVI_g(img):
    if len(img.shape) == 3:
        return 1.0 * ((img[:, :, 3]-img[:, :, 1]) / (img[:, :, 3] + img[:, :, 1]))
    elif len(img.shape) == 4:
        return 1.0 * ((img[:, :, 3, :]-img[:, :, 1, :]) / (img[:, :, 3, :] + img[:, :, 1, :]))
    elif len(img.shape)==2:
        return 1.0 * ((img[:, 3]-img[:,1]) / (img[:, 3] + img[:, 1]))

def NDVI_b(img):
    if len(img.shape) == 3:
        return 1.0 * ((img[:, :, 3]-img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0]))
    elif len(img.shape) == 4:
        return 1.0 * ((img[:, :, 3, :]-img[:, :, 0, :]) / (img[:, :, 3, :] + img[:, :, 0, :]))
    elif len(img.shape)==2:
        return 1.0 * ((img[:, 3]-img[:, 0]) / (img[:, 3] + img[:, 0]))

def NDWI(img):
    if len(img.shape) == 3:
        return 1.0 * ((img[:, :, 1]-img[:, :, 3]) / (img[:, :, 1]+img[:, :, 3]))
    elif len(img.shape) == 4:
        return 1.0 * ((img[:, :, 1, :]-img[:, :, 3, :]) / (img[:, :, 1, :]+img[:, :, 3, :]))
    elif len(img.shape) == 2:
        return 1.0 * ((img[:, 1]-img[:, 3]) / (img[:, 1]+img[:, 3]))

def EVI(img):
    if len(img.shape) == 3:
        return 2.5 * ((img[:, :, 3]-img[:, :, 2]) /
                  (img[:, :, 3]+6*img[:, :, 2] - 7.5 * img[:, :, 0] + 1))
    elif len(img.shape) == 4:
        return 2.5 * ((img[:, :, 3, :]-img[:, :, 2, :]) /
                  (img[:, :, 3, :]+6*img[:, :, 2, :] - 7.5 * img[:, :, 0, :] + 1))
    elif len(img.shape) == 2:
        return 2.5 * ((img[:, 3]-img[:, 2]) /
                  (img[:, 3]+6*img[:, 2] - 7.5 * img[:, 0] + 1))

def SAVI(img):
    if len(img.shape) == 3:
        return ((img[:, :, 3]-img[:, :, 2]) / (img[:, :, 3]+img[:, :, 2]+0.5)) * 1.5
    if len(img.shape) == 4:
        return ((img[:, :, 3, :]-img[:, :, 2, :]) / (img[:, :, 3, :]+img[:, :, 2, :]+0.5)) * 1.5
    if len(img.shape) == 2:
        return ((img[:,  3]-img[:, 2]) / (img[:, 3]+img[:, 2]+0.5)) * 1.5

def MSAVI(img):
    if len(img.shape) == 3:
        return (2*img[:, :, 3] + 1 -
                np.sqrt((2 * img[:, :, 3] + 1)**2 -
                        8 * (img[:, :, 3] - img[:, :, 2]))) / 2.0
    if len(img.shape) == 4:
        return (2*img[:, :, 3, :] + 1 -
                np.sqrt((2 * img[:, :, 3, :] + 1)**2 -
                        8 * (img[:, :, 3, :] - img[:, :, 2, :]))) / 2.0
    if len(img.shape) == 2:
        return (2*img[:, 3] + 1 -
                np.sqrt((2 * img[:, 3] + 1)**2 -
                        8 * (img[:, 3] - img[:, 2]))) / 2.0

################################################################################
# Create dictionary of image features per segment

def get_superpixel_image_features(superpixel):
    '''
    DESC: Get image features per segment
    INPUT: image=np.array(), segments=np.array(), plot=bool
    -----
    OUTPUT: zone_dict of image features per segment/superpixel
    '''
    zone_dict= {}
    NDVI_r_img = NDVI_r(superpixel)
    NDVI_g_img = NDVI_g(superpixel)
    NDVI_b_img = NDVI_b(superpixel)
    NDWI_img = NDWI(superpixel)
    EVI_img = EVI(superpixel)
    SAVI_img = SAVI(superpixel)
    MSAVI_img = MSAVI(superpixel)

    channels_min = np.nanmin(superpixel, axis=(0))
    channels_max = np.nanmax(superpixel, axis=(0))
    channels_mean = np.nanmean(superpixel, axis=(0))
    channels_std = np.nanstd(superpixel, axis=(0))
    channels_median = np.nanmedian(superpixel, axis=(0))

    zone_dict["blue"]=(channels_min[0], channels_max[0], channels_std[0], channels_mean[0], channels_median[0])
    zone_dict["green"]=(channels_min[1], channels_max[1], channels_std[1], channels_mean[1], channels_median[1])
    zone_dict["red"]=(channels_min[2], channels_max[2], channels_std[2], channels_mean[2], channels_median[2])
    zone_dict["nir"]=(channels_min[3], channels_max[3], channels_std[3], channels_mean[3], channels_median[3])

    zone_dict["NDVI_r"]=(np.nanmin(NDVI_r_img), np.nanmax(NDVI_r_img), np.nanstd(NDVI_r_img), np.nanmean(NDVI_r_img), np.nanmedian(NDVI_r_img))
    zone_dict["NDVI_g"]=(np.nanmin(NDVI_g_img), np.nanmax(NDVI_g_img), np.nanstd(NDVI_g_img), np.nanmean(NDVI_g_img), np.nanmedian(NDVI_g_img))
    zone_dict["NDVI_b"]=(np.nanmin(NDVI_b_img), np.nanmax(NDVI_b_img), np.nanstd(NDVI_b_img), np.nanmean(NDVI_b_img), np.nanmedian(NDVI_b_img))
    zone_dict["EVI"]=(np.nanmin(EVI_img), np.nanmax(EVI_img), np.nanstd(EVI_img), np.nanmean(EVI_img), np.nanmedian(EVI_img))
    zone_dict["SAVI"]=(np.nanmin(SAVI_img), np.nanmax(SAVI_img), np.nanstd(SAVI_img), np.nanmean(SAVI_img), np.nanmedian(SAVI_img))
    zone_dict["MSAVI"]=(np.nanmin(MSAVI_img), np.nanmax(MSAVI_img), np.nanstd(MSAVI_img), np.nanmean(MSAVI_img), np.nanmedian(MSAVI_img))
    zone_dict["NDWI"]=(np.nanmin(NDWI_img), np.nanmax(NDWI_img), np.nanstd(NDWI_img), np.nanmean(NDWI_img), np.nanmedian(NDWI_img))

    return zone_dict

################################################################################
# Get region properties per segments

def get_region_props(image, segments, plot=False):
    '''
    DESC: Get segment/region properties per segment
    INPUT: image=np.array(), segments=np.array(), plot=bool
    -----
    OUTPUT: seg_stats=dictionary key is image filename, value segment properties,
            g=Networkx graph
    '''
    grayscaledimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
    regions_gray = regionprops(segments, grayscaledimg)
    regions_blue = regionprops(segments, image[:,:,0])
    regions_green = regionprops(segments, image[:,:,1])
    regions_red = regionprops(segments, image[:,:,2])
    seg_stats = {'gray':regions_gray,
                'red': regions_red,
                'green':regions_green,
                'blue':regions_blue}
    # Calculate simiarity of segments and graph

    if plot:
        mean_label_rgb = get_mean_pixel_value(image, segments, plot=plot)
        g = graph.rag_mean_color(image, segments, mode='similarity')
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
        ax[0].set_title('RAG drawn with default settings')
        lc = graph.show_rag(segments, g, image, edge_cmap='viridis',ax=ax[0])
    # specify the fraction of the plot area that will be used to draw the colorbar
        fig.colorbar(lc, fraction=0.03, ax=ax[0])
        ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
        lc = graph.show_rag(segments, g, image,
                            img_cmap='gray', edge_cmap='viridis', ax=ax[1])
        fig.colorbar(lc, fraction=0.03, ax=ax[1])
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

        for region in regions_gray:
            g.node[region['label']]['centroid'] = region['centroid']
        edges_drawn_all, weights = display_edges(mean_label_rgb, rag=g, threshold=np.inf )
        plt.imshow(edges_drawn_all)
        return seg_stats, g
    else:
        return seg_stats

################################################################################
# Generating feature dictionary functions

def get_superpixels(image, segments):
    '''
    DESC: Get flattened array/list of segment and pixel values
    INPUT: image=np.array(), segments=np.array()
    -----
    OUTPUT: flattened array
    '''
    superpixel_list = sp_idx(segments)
    superpixel = [image[idx] for idx in superpixel_list]
    return superpixel


def get_image_colorfulness(image):
    # Get how colorful image is
    C = image_colorfulness(image[:,:,:3])
    return C


def get_segment_masks(image, segments):
    '''
    DESC: Get segment mask
    INPUT: image=np.array(), segments=np.array()
    -----
    OUTPUT: mask
    '''
    # Segment masks
    masks=[]
    for (i, seg) in enumerate(np.unique(segments)):
    # construct a mask for the segment

        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == seg] = 255
        masks.append(mask)

    # show the masked region (this will crash notebook)
#         cv2.imshow("Mask", mask)
#         cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
    return mask


def get_mean_pixel_value(image, segments, plot=False):
    # Get avg pixel value of segment
    mean_label_rgb = color.label2rgb(segments, image, kind='avg')
    if plot:
        plt.imshow(mean_label_rgb)
    return mean_label_rgb


def replace_inf_by_nan(img):
    img[img == np.inf] = np.nan
    img[img == -np.inf] = np.nan
    return img


def get_segment_colorfulness(image, segments, plot=False):
    # loop over each of the unique superpixels
    seg_color = {}
    vis = np.zeros(image.shape[:2], dtype=np.float32)

    for v in np.unique(segments):
        # construct a mask for the segment so we can compute image statistics for *only* the masked region
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0
        # compute the superpixel colorfulness, then update the visualization array
        C_seg = segment_colorfulness(image[:,:,:3], mask)
        vis[segments == v] = C_seg
        seg_color[v] = C_seg

    if plot:
        vis = rescale_intensity(vis, out_range=(0, 255)).astype('uint8')
        # overlay the superpixel colorfulness visualization on the original image
        alpha = 0.3
        overlay = np.dstack([vis] * 3)
        output = image.copy().astype('uint8')
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # show the output images (this will crash notebook)
        #     cv2.imshow("Input", image)
        #     cv2.imshow("Visualization", vis)
        #     cv2.imshow("Output", output)
    return seg_color


def get_threshold_image(image, segments, rag, weights, plot=False):
    thres_labels = graph.cut_threshold(segments, rag, np.percentile(np.array(weights), 15))
    thres_label_rgb = color.label2rgb(thres_labels, image, kind='avg')
    if plot:
        plt.imshow(thres_label_rgb)
        plt.show()
    return thres_label_rgb

###############################################################################
# Generate features

def get_segment_features(images, segments):
    '''
    DESC: Get dictionaies of generated features
    INPUT: images=np.array(), segments=np.array()
    -----
    OUTPUT: feature dictionaries with key as image filename
    '''
    superpixels, features, segment_properties, rags, image_color, seg_color ={},{},{},{},{},{}
    for k in images.keys():
        superpixels[k] = get_superpixels(images[k], segments[k])
        segment_properties[k] = get_region_props(images[k][:,:,:3], segments[k])
        image_color[k] = get_image_colorfulness(images[k])
        seg_color[k] = get_segment_colorfulness(images[k], segments[k])
        features[k] = {}
        for ind, spxl in enumerate(superpixels[k]):
            seg = ind + 1
            features[k][seg]= get_superpixel_image_features(spxl)
    return features, segment_properties, image_color, seg_color

###############################################################################
# Generate df Helpher functions

def get_image_features_df(features):
    '''
    DESC: create dataframe of image features per segment
    INPUT: features=dict key is image filename, values are image features
    -----
    OUTPUT: df
    '''
    files,frames=[],[]
    for f, seg in features.items():
        files.append(f)
        frames.append(pd.DataFrame.from_dict(seg, orient='index'))

    f_df = pd.concat(frames, keys=files)
    orig_cols = f_df.columns
    f_df.reset_index(inplace=True)
    f_df.rename(columns={'level_0':'filename', 'level_1':'segment'},inplace=True)

    f_df[['red_min','red_max','red_std','red_mean','red_median']] = f_df['red'].apply(pd.Series)
    f_df[['blue_min','blue_max','blue_std','blue_mean','blue_median']] = f_df['blue'].apply(pd.Series)
    f_df[['green_min','green_max','green_std','green_mean','green_median']] = f_df['green'].apply(pd.Series)
    f_df[['nir_min','nir_max','nir_std','nir_mean','nir_median']] = f_df['nir'].apply(pd.Series)

    f_df[['SAVI_min','SAVI_max','SAVI_std','SAVI_mean','SAVI_median']] = f_df['SAVI'].apply(pd.Series)
    f_df[['NDVI_b_min','NDVI_b_max','NDVI_b_std','NDVI_b_mean','NDVI_b_median']] = f_df['NDVI_b'].apply(pd.Series)
    f_df[['NDVI_g_min','NDVI_g_max','NDVI_g_std','NDVI_g_mean','NDVI_g_median']] = f_df['NDVI_g'].apply(pd.Series)
    f_df[['NDVI_r_min','NDVI_r_max','NDVI_r_std','NDVI_r_mean','NDVI_r_median']] = f_df['NDVI_r'].apply(pd.Series)
    f_df[['EVI_min','EVI_max','EVI_std','EVI_mean','EVI_median']] = f_df['EVI'].apply(pd.Series)
    f_df[['MSAVI_min','MSAVI_max','MSAVI_std','MSAVI_mean','MSAVI_median']] = f_df['MSAVI'].apply(pd.Series)
    f_df[['NDWI_min','NDWI_max','NDWI_std','NDWI_mean','NDWI_median']] = f_df['NDWI'].apply(pd.Series)

    f_df.drop(orig_cols, axis=1, inplace=True)
    return f_df

def get_seg_color_df(seg_color):
    '''
    DESC: Get df of segment colorfulness
    INPUT: seg_color=dict key is image filename, value is colorfulness of segment
    -----
    OUTPUT: flattened array
    '''
    files,frames=[],[]
    for f, seg in seg_color.items():
        files.append(f)
        frames.append(pd.DataFrame.from_dict(seg, orient='index'))

    c_df = pd.concat(frames, keys=files)
    c_df.reset_index(inplace=True)
    c_df.rename(columns={'level_0':'filename', 'level_1':'segment', 0:'seg_colorfulness'},inplace=True)
    return c_df

def get_rag_properties_df(rags):
    '''
    DESC: Get df of region adjecenty graph per image
    INPUT: rags=dict key is image filename, value is RAG properties
    -----
    OUTPUT: df
    '''
    rag_dfs = []
    for k in rags.keys():
        l=list(rags[k].node(data=True))
        q ={}
        for b in l:
            q[b[0]] = b[1]
        rag_df = pd.DataFrame.from_dict(q, orient='index')
        rag_df[['blue_total_color','green_total_color','red_total_color']] = rag_df['total color'].apply(pd.Series)
        rag_df['segment'] = rag_df['labels'].apply(lambda x: int(x[0]))
        rag_df[['blue_mean_color','green_mean_color','red_mean_color']] = rag_df['mean color'].apply(pd.Series)
        rag_df.drop(['centroid', 'total color', 'mean color', 'labels'], axis=1, inplace=True)
        rag_df.rename(columns={'pixel count':'pixel_count'}, inplace=True)
        rag_df['filename'] = k
        rag_dfs.append(rag_df)
    return pd.concat(rag_dfs)

def get_segment_properties_df(segment_properties):
    '''
    DESC: Get df of segment region properties
    INPUT: rags=dict key is image filename, value is segment properties http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    -----
    OUTPUT: df
    '''
    image_dfs = []
    for image in segment_properties.keys():
        for segment in range(len(segment_properties[image]['gray'])):
            image_df = pd.DataFrame({
                                      'area':segment_properties[image]['gray'][segment]['area'],
                                      'bbox':[segment_properties[image]['gray'][segment]['bbox']],
                                      'bbox_area':segment_properties[image]['gray'][segment]['bbox_area'],
                                      'centroid_x':segment_properties[image]['gray'][segment]['centroid'][1],
                                    'centroid_y':segment_properties[image]['gray'][segment]['centroid'][0],
                                      'convex_area':segment_properties[image]['gray'][segment]['convex_area'],
    #                                   'convex_image':seg_stats_15[image]['gray'][segment]['convex_image'],
                                        'coord':[segment_properties[image]['gray'][segment]['coords']],
    #                                   'coord_x':seg_stats_15[image]['gray'][segment]['coords'][0],
    #                                 'coord_y':seg_stats_15[image]['gray'][segment]['coords'][1],
                                      'eccentricity':segment_properties[image]['gray'][segment]['eccentricity'],
                                      'equivalent_diameter':segment_properties[image]['gray'][segment]['equivalent_diameter'],
                                      'euler_number':segment_properties[image]['gray'][segment]['euler_number'],
                                      'extent':segment_properties[image]['gray'][segment]['extent'],
                                      'filled_area':segment_properties[image]['gray'][segment]['filled_area'],
    #                                   'filled_image':seg_stats_15[image]['gray'][segment]['filled_image'],
    #                                   'image':seg_stats_15[image]['gray'][segment]['image'],
                                      'inertia_tensor':[segment_properties[image]['gray'][segment]['inertia_tensor']],
                                        'inertia_tensor_eigvals':[[segment_properties[image]['gray'][segment]['inertia_tensor_eigvals']]],
    #                                   'intensity_image':seg_stats_15[image]['gray'][segment]['intensity_image'],
                                      'label':segment_properties[image]['gray'][segment]['label'],
#                                         'local_centroid_':[segment_properties[image]['gray'][segment]['local_centroid']],
                                      'local_centroid_x':segment_properties[image]['gray'][segment]['local_centroid'][1],
                                        'local_centroid_y':segment_properties[image]['gray'][segment]['local_centroid'][0],
                                      'major_axis_length':segment_properties[image]['gray'][segment]['major_axis_length'],
                                      'gray_max_intensity':segment_properties[image]['gray'][segment]['max_intensity'],
                                      'gray_mean_intensity':segment_properties[image]['gray'][segment]['mean_intensity'],
                                      'gray_min_intensity':segment_properties[image]['gray'][segment]['min_intensity'],
                                      'red_max_intensity':segment_properties[image]['red'][segment]['max_intensity'],
                                      'red_mean_intensity':segment_properties[image]['red'][segment]['mean_intensity'],
                                      'red_min_intensity':segment_properties[image]['red'][segment]['min_intensity'],
                                      'blue_max_intensity':segment_properties[image]['blue'][segment]['max_intensity'],
                                      'blue_mean_intensity':segment_properties[image]['blue'][segment]['mean_intensity'],
                                      'blue_min_intensity':segment_properties[image]['blue'][segment]['min_intensity'],
                                      'green_max_intensity':segment_properties[image]['green'][segment]['max_intensity'],
                                      'green_mean_intensity':segment_properties[image]['green'][segment]['mean_intensity'],
                                      'green_min_intensity':segment_properties[image]['green'][segment]['min_intensity'],
                                      'minor_axis_length':segment_properties[image]['gray'][segment]['minor_axis_length'],
                                      'moments':[segment_properties[image]['gray'][segment]['moments']],
    #                                     'moments_y':[seg_stats_15[image]['gray'][segment]['moments'][:,1]],
    #                                     'moments_z':[seg_stats_15[image]['gray'][segment]['moments'][:,2]],
                                      'moments_central':[segment_properties[image]['gray'][segment]['moments_central']],
    #                                     'moments_central_y':[seg_stats_15[image]['gray'][segment]['moments_central'][:,1]],
    #                                     'moments_central_z':[seg_stats_15[image]['gray'][segment]['moments_central'][:,2]],
                                      'moments_hu':[segment_properties[image]['gray'][segment]['moments_hu']],
                                      'moments_normalized':[segment_properties[image]['gray'][segment]['moments_normalized']],
                                      'orientation':segment_properties[image]['gray'][segment]['orientation'],
                                      'perimeter':segment_properties[image]['gray'][segment]['perimeter'],
                                      'solidity':segment_properties[image]['gray'][segment]['solidity'],
                                      'weighted_centroid':[segment_properties[image]['gray'][segment]['weighted_centroid']],
                                      'weighted_local_centroid':[segment_properties[image]['gray'][segment]['weighted_local_centroid']],
                                      'weighted_moments':[segment_properties[image]['gray'][segment]['weighted_moments']],
                                      'weighted_moments_central':[segment_properties[image]['gray'][segment]['weighted_moments_central']],
                                      'weighted_moments_hu':[segment_properties[image]['gray'][segment]['weighted_moments_hu']]
                                    }, index=[0])
#             image_df[['moments_1', 'moments_2','moments_3']] = image_df['moments'].apply(pd.Series)
#             image_df[['moments_central_1', 'moments_central_2','moments_central_3']] = image_df['moments_central'].apply(pd.Series)
#             image_df[['moments_hu_1', 'moments_hu_2','moments_hu_3']] = image_df['moments_central'].apply(pd.Series)
            segment = segment + 1
            image_df['segment'] = segment
            image_df['filename'] = image.strip('\n')
            image_dfs.append(image_df)
    return pd.concat(image_dfs)


################################################################################
# Relabeling segments by gray_mean_intensity

def relabel(df):
    '''
    DESC: Relabel segments by gray_mean_intensity
    INPUT: df=df
    -----
    OUTPUT: df with relabled_segs col
    '''
    df.sort_values(['gray_mean_intensity'],ascending=False,inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['relabeled_segs'] = df.index+1
    del df['filename']
    return df

################################################################################
# Create df from feature dictionaries with key as image filename

def create_df(images_df, griffin_features, features, segment_properties, image_color, seg_color, original_segs):
    '''
    DESC: Create single dataframe appended to images df from feature generation dictionaries (outputs from get_segment_features)
    INPUT: images_df=df, griffin_features=df (from extractFeatures), features - seg_color=dicts (from get_segment_features), original_segs=dict ()
    -----
    OUTPUT: concatened df
    '''
    result_df_list =[]
    # generating df
    f_df = get_image_features_df(features)
    c_df = get_seg_color_df(seg_color)
    # rag_df = get_rag_properties_df(rags)
    f_df = pd.merge(f_df,c_df, on=['filename', 'segment'], how='left')
    seg_props_df = get_segment_properties_df(segment_properties)
    seg_c_f_df = pd.merge(f_df, seg_props_df, on=['filename', 'segment'])
    # rag_seg_c_f_df = pd.merge(seg_c_f_df, rag_df, on=['filename', 'segment'])
    grif_seg_c_f_df = pd.merge(seg_c_f_df, griffin_features, on=['filename', 'segment'])

    relabel_df = grif_seg_c_f_df.groupby(['filename']).apply(relabel)
    relabel_df.reset_index(inplace=True)
    del relabel_df['level_1']


    for c in relabel_df.columns:
        if c not in ['filename', 'relabeled_segs']:
            pivot_df = relabel_df.pivot(values = c,index='filename',columns='relabeled_segs')
            pivot_df= pivot_df.add_prefix(str(c)+'_seg'+'_')
            result_df_list.append(pivot_df)
    result_df = pd.concat(result_df_list,axis=1)
    result_df.reset_index(inplace=True)

    orig_segs = pd.DataFrame.from_dict(original_segs, orient='index').reset_index()
    orig_segs.rename(columns={'index':'filename',0:'num_orig_segments'}, inplace=True)

    img_c_df = pd.DataFrame.from_dict(image_color, orient='index').reset_index()
    img_c_df.rename(columns={'index':'filename',0:'img_colorfulness'}, inplace=True)

    img_c_df = pd.merge(img_c_df, orig_segs, on=['filename'])
    final_df = pd.merge(result_df, img_c_df, on=['filename'], how='left')
    final_df = pd.merge(images_df,final_df, on=['filename'], how='inner')

    return final_df

################################################################################
# multiprocessing

def get_batch(df, col, batchsize=50, save=True):
    '''
    DESC: create batches of field IDs
    INPUT: df=df, col=str() [col for unique identifier (ex fieldID)], batchsize=int()
    -----
    OUTPUT: list of unique batched ids by col value
    '''
    ls = []
    for i in range(int(2200/batchsize)):
        n = i*batchsize
        k = (i+1)*batchsize
        f = df[col].unique().tolist()[n:k]
        if save:
            save_obj(f, 'fields{}_{}.p'.format(n,k))
        ls.append(f)
    return ls


def batch_iterator(n_items, batch_size):
    import math
    n_batches = int(math.ceil(n_items/(batch_size+1e-9)))
    for b in range(n_batches):
        start = (b*batch_size)
        end = ((b+1)*batch_size)
        if end >= n_items:
            end = n_items
        yield (start, end)


################################################################################
# Plotting functions

def display_edges(image, rag, threshold):
    """Draw edges of a RAG on its image

    Returns a modified image with the edges drawn.Edges are drawn in green
    and nodes are drawn in yellow.

    Parameters
    ----------
    image : ndarray
        The image to be drawn on.
    g : RAG
        The Region Adjacency Graph.
    threshold : float
        Only edges in `g` below `threshold` are drawn.

    Returns:
    out: ndarray
        Image with the edges drawn.
    """
    image = image.copy()
    rag2 = rag.copy()
    weights = []
    for edge in rag2.edges():
        n1, n2 = edge

        r1, c1 = map(int, rag2.node[n1]['centroid'])
        r2, c2 = map(int, rag2.node[n2]['centroid'])

        line  = draw.line(r1, c1, r2, c2)
        circle = draw.circle(r1,c1,2)

        if rag2[n1][n2]['weight'] < threshold:
            image[line] = 0,1,0
            weights.append(rag2[n1][n2]['weight'])
        image[circle] = 1,1,0

    return image, weights


def plot_segments(image, seg):
    '''
    DESC: Plot segments boundaries, centroids on image and mean color per segment image
    INPUT: image=np.array(), seg=np.array()
    -----
    OUTPUT: mean color plot, segment boundary with labeled centroid
    '''
    out = color.label2rgb(seg, image, kind='avg')
    out = segmentation.mark_boundaries(out, seg, (0, 0, 0))
    io.imshow(out)
    io.show()
    grayscaledimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    regions = regionprops(seg, grayscaledimg)
    relabeled_centroids = [(region['label'], region['centroid']) for region in regions]
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), seg))
    c_dict = {}
    for label, center in relabeled_centroids:
        x,y =center
        c_dict[label] = center
        plt.text(y, x, label, color='yellow')
    plt.show()
    return
