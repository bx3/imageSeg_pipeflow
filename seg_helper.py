import numpy as np
from skimage.measure import label
import math

from scipy.misc import imresize
from skimage import morphology

import matplotlib.pyplot as plt

from skimage import color

from PIL import Image
from scipy.ndimage import imread

import matplotlib.pyplot as plt
import skimage.measure as sk_measure

from skimage.segmentation import felzenszwalb
import cv2
from growcut import growcut_cy
import pymeanshift as pms
from skimage.segmentation import random_walker

import time

def get_rid_of_black_frame(image):
    m,n,_ = np.shape(image)
    l,r,u,d = 0,n,0,m
    for i in range(m):
        if np.sum(image[i,math.floor(n/2),:]) == 0:
            u = i
        else:
            break
    for i in reversed(range(m)):
        if np.sum(image[i,math.floor(n/2),:]) == 0:
            d = i
        else:
            break
    for i in range(n):
        if np.sum(image[math.floor(m/2),i,:]) == 0:
            l = i
        else:
            break
    for i in reversed(range(n)):
        if np.sum(image[math.floor(m/2),i,:]) == 0:
            r = i
        else:
            break
    return image[u:d, l:r],u,d,l,r
    
def normalize_rgb(rgb_img):
    rgb_img_f = np.copy(rgb_img)*1.0
    rgb_sum_img_f = rgb_img_f[:,:,0]+rgb_img_f[:,:,1]+rgb_img_f[:,:,2]
    n_r = (rgb_img_f[:,:,0]+1)/(rgb_sum_img_f+3)
    n_g = (rgb_img_f[:,:,1]+1)/(rgb_sum_img_f+3)
    n_b = (rgb_img_f[:,:,2]+1)/(rgb_sum_img_f+3)

    n_rgb_img = np.zeros_like(rgb_img_f)
    n_rgb_img[:,:,0] = n_r
    n_rgb_img[:,:,1] = n_g
    n_rgb_img[:,:,2] = n_b
    return n_rgb_img
    
# use for putting mask on original image
def mask_rgb(mask,rgb_img):
    r = rgb_img[:,:,0]
    g = rgb_img[:,:,1]
    b = rgb_img[:,:,2]
    m_img = np.zeros_like(rgb_img)
    m_img[:,:,0] = r*mask
    m_img[:,:,1] = g*mask
    m_img[:,:,2] = b*mask
    return m_img
    
def compute_region_mean(mask, mono_image):
    num_back_pixel = np.count_nonzero(mask)
    crop_local_back = mono_image*(mask)
    mean_crop_local_back = np.sum(crop_local_back)/num_back_pixel
    return mean_crop_local_back
    
# accept only 2d image
def compute_region_std(mask, mono_image):
    m,n = np.shape(mask)
    masked_image = mask* mono_image
    back_i, back_j = np.nonzero(masked_image)    
    back_pixels_list = []
    for k in range(len(back_i)):        
        back_pixels_list.append(masked_image[back_i[k],back_j[k]])
    back_pixels= np.array(back_pixels_list)
    return np.std(back_pixels)            
    
def put_back_boundary(old_mask, new_mask, length):    
    hallow_mask = np.copy(old_mask)
    hallow_mask[length:-length,length:-length] = 1
    sum_mask = np.logical_and(hallow_mask, new_mask)
    return sum_mask    
    
def identify_if_has_hole(mask, ratio, image, r_m, r_v, b_m, b_v):  
    # this label_num exclude background
    labeled_image, label_num = label(mask==0, neighbors=None, background=None, return_num=True, connectivity=2)
    m,n = np.shape(mask)
    
    back_id = labeled_image[0,0]    
    back_mask = labeled_image == back_id
    
    hole_id_list = [i for i in range(1,label_num+1)]    
    hole_id_list.remove(back_id)        
    
    hole_seg = np.logical_and(labeled_image!=back_id, labeled_image)*labeled_image    
    # if hole is too small, ignore
    for i in hole_id_list:        
        hole_pixel_num = np.sum(hole_seg==i)        
        if hole_pixel_num < math.ceil(m/ratio)*math.ceil(n/ratio):
            hole_id_list.remove(i)
            hole_seg = np.logical_and(hole_seg, hole_seg!=i)*hole_seg            
    #if r b component is too large
    #for i in hole_id_list:        
    #    hole_r_mean = compute_region_mean(image[:,:,0], hole_seg==i)
    #    hole_b_mean = compute_region_mean(image[:,:,2], hole_seg==i)
    #    if hole_r_mean > r_m + r_v or hole_b_mean > b_m + b_v:
    #        hole_id_list.remove(i)
    #        hole_seg = np.logical_and(hole_seg, hole_seg!=i)*hole_seg                
    
    
    return hole_id_list, hole_seg#, back_mask          
    
def crop_image(object_mask, rgb_img, extend):
    # crop image
    p_u, p_d = 0,0
    p_l, p_r = 0,0
    m,n = np.shape(object_mask)
    for i in range(m):
        if 1 in object_mask[i,:]:
            p_d = i + 1

    for i in reversed(range(m)):
        if 1 in object_mask[i,:]:
            p_u = i

    for i in (range(n)):
        if 1 in object_mask[:,i]:
            p_r = i + 1

    for i in reversed(range(n)):
        if 1 in object_mask[:,i]:
            p_l = i     
            
    crop_u = max(p_u-extend,0)
    crop_d = min(p_d+extend,m)
    crop_l = max(p_l-extend,0)
    crop_r = min(p_r+extend,n)
    crop_local_mask = object_mask[crop_u:crop_d, crop_l:crop_r]
    crop_local_image = rgb_img[crop_u:crop_d, crop_l:crop_r]
    
    return crop_local_mask, crop_local_image, crop_u,crop_d, crop_l, crop_r
    
def identify_background(seg_image, id_segs_dict, boundary_ratio):
    # use  local info
    m,n = np.shape(seg_image)
    d = math.ceil(m*boundary_ratio)
    
    back_point_list = [[d,d],[d,n-d],[m-d,d],[m-d,n-d]]    
    
    back_ids = []
    # use center property to identify background
    for p in back_point_list:        
        back_ids.append(seg_image[p[0],p[1]])
            
    unique_back_ids = np.unique(np.array(back_ids))
    
    if len(unique_back_ids) ==1 :
        return seg_image, unique_back_ids[0]
    else:
        new_id = unique_back_ids[0]
        for i in list(unique_back_ids[1:]):
            seg_image = seg_image - id_segs_dict[i]*i+new_id*id_segs_dict[i]
            id_segs_dict[new_id] = (new_id==seg_image)*1  
            del id_segs_dict[i]            
        
        return seg_image,new_id                                        
        
def pad_boundary(seg_mask, length,value):
    m,n = np.shape(seg_mask)
    width = math.floor(length/2)
    seg_mask[:width,:] = value
    seg_mask[(m-width):,:] = value   
    seg_mask[:,:width] = value
    seg_mask[:,(n-width):] = value    
    
# old metrics
def old_metrics(mask, ground):
    error = np.count_nonzero(np.logical_xor(mask, ground)) / np.count_nonzero(np.logical_or(mask, ground)) * 100
    accuracy = 100-error
    print('old_metric_result', accuracy)    
    
def compute_confusion_matrix(mask,ground):
    mask_f = mask.flatten()*1.0
    ground_f = ground.flatten()*1.0
    C = confusion_matrix(mask_f, ground_f)
    print(C)
    accuracy = (C[0,0]+C[1,1])/np.sum(C)
    print('accuracy', accuracy)
    precision = (C[0,0])/(C[0,0]+C[0,1])
    print('precision', precision)
    recall = (C[0,0])/(C[0,0]+C[1,0])
    print('recall', recall)    
    F = 2*precision*recall/(precision+recall)
    print('F score', F)
    
def put_back_cropped_mask(crop_u,crop_d, crop_l,crop_r, final_c_clustered_mask, m,n):
    obj_detail_mask = np.zeros((m,n))
    obj_detail_mask[crop_u:crop_d, crop_l:crop_r] = final_c_clustered_mask
    return obj_detail_mask
 
def resize_back_to_ori(sampling_ratio, obj_detail_mask, ori_m, ori_n): 
    if sampling_ratio != 1:
        obj_detail_mask = imresize(obj_detail_mask,sampling_ratio*100, 'nearest')        
    obj_detail_mask = obj_detail_mask[:ori_m,:ori_n]>0
    return obj_detail_mask
    
def close_image_with_back(fore_mask, close_length):    
    return morphology.binary_closing(fore_mask==0, np.ones((close_length,close_length)))==0
    
# not used
def clear_black_edges(back_id, id_segs_dict, threshold):
    back_mask = id_segs_dict[back_id]
    for the_id, mask in id_segs_dict.items():        
        #imgplot = plt.imshow(mask)
        #plt.show()
        region_perimeter = perimeter(mask, neighbourhood=4)
        region_area = np.sum(mask)
        a_over_p = region_area/region_perimeter
        print(a_over_p)
        if a_over_p < threshold:
            back_mask = back_mask + mask
    return back_mask

def plot_mask_image(clustered_mask, crop_local_image):  
    plt.imshow(clustered_mask)
    plt.title('clustered mask')
    plt.show()

    plt.imshow(mask_rgb(clustered_mask, crop_local_image))
    plt.title('clustered mask')
    plt.show()

    plt.imshow(crop_local_image)
    plt.title('clustered mask')
    plt.show()

# change feature here
def compute_region_color_mean(mask, color_image):
    num_back_pixel = np.count_nonzero(mask)
    c_mask = np.dstack((mask,mask,mask))
    crop_local_back = color_image * c_mask    
    mean_c1 = np.sum(crop_local_back[:,:,0])/num_back_pixel
    mean_c2 = np.sum(crop_local_back[:,:,1])/num_back_pixel
    mean_c3 = np.sum(crop_local_back[:,:,2])/num_back_pixel

    feature = [mean_c1, mean_c2,mean_c3]
    
    return feature

def compute_region_color_std(mask, color_image):
    m,n = np.shape(mask)        
        
    masked_image = color_image * np.dstack((mask,mask,mask))
    
    i,j = np.nonzero(mask)       
    a_list,b_list,c_list = [],[],[]        
    
    for k in range(len(i)):        
        a_list.append(masked_image[i[k],j[k],0])        
        b_list.append(masked_image[i[k],j[k],1])
        c_list.append(masked_image[i[k],j[k],2])
    a_pixels= np.array(a_list)
    b_pixels= np.array(b_list)
    c_pixels= np.array(c_list)

    feat_std = [np.std(a_pixels), np.std(b_pixels), np.std(c_pixels)]
    
    return feat_std   

# consider regions which has pixel within extend t to be background
def get_back_labels(labels_image, extend):
    l = list(np.unique(labels_image[:,:extend]))
    r = list(np.unique(labels_image[:,-extend:]))
    u = list(np.unique(labels_image[:extend,:]))
    d = list(np.unique(labels_image[-extend:,:]))
        
    back_list = np.array(l+r+u+d)
    
    back_labels = np.unique(back_list)
    return back_labels    
    
def make_color_generator(NUM_COLORS):
    cm = pylab.get_cmap('gist_rainbow')
    color_list = []
    for i in range(NUM_COLORS):
        color_list.append(cm(1.*i/NUM_COLORS))  # color will now be an RGBA tuple
    
    shuffle(color_list)
        
    cgen = (i for i in color_list)
    return cgen

def paint_graph_by_community(G, partition, pos=None, point_size=100):
    community = partition.values()
    if pos is None:
        pos = nx.spring_layout(G)
    num_com = len(community)
    color_gen = make_color_generator(num_com)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]        
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = point_size,
                                    node_color = next(color_gen), with_labels=True)
        
    nx.draw_networkx_edges(G,pos, alpha=0.5)
    nx.draw_networkx_labels(G,pos)
    plt.show()        
    
def paint_regions_from_index_list(fore_region_list, labels_image):
    canvas = np.zeros_like(labels_image)
    for i in fore_region_list:
        mask = (i == labels_image)        
        canvas = canvas + mask
        canvas = np.logical_or(canvas, mask)
    return canvas
    
def construct_region_mask_dict(index_list, labels_image):
    index_list = list(index_list)
    region_mask_dict = {}
    for i in index_list:
        region_mask_dict[i] = (i==labels_image)
    return region_mask_dict    
    
def get_object_mask_from_connected_regions(fore_mask, neighbors=None):
    labeled_image, label_num = sk_measure.label(fore_mask, neighbors=neighbors, background=0, return_num=True, connectivity=None)
    if label_num == 1:
        object_mask = fore_mask
    else:
        sum_list = np.zeros(label_num)    
        for i in range(1,label_num+1):        
            sum_list[i-1] = np.sum(labeled_image==i)        
            
        obj_id = np.argmax(sum_list)+1
        object_mask = labeled_image==obj_id
        
    return labeled_image, object_mask
    
#sampling point
def get_sample_point(image,num, mask):
    m,n = np.shape(mask)
    
    points_list = []    
    for i in range(m):
        for j in range(n):            
            if mask[i,j]==1:                
                r = image[i,j,0]
                g = image[i,j,1]
                b = image[i,j,2]
                p = np.array([r,g,b,i,j])
                points_list.append(p)
            
    sampled_point = np.random.randint(0,len(points_list)-1, num)    
    points = [points_list[i] for i in sampled_point]        
    return points
    
#cluster image
def cluster_image(X,centroids,thresh, is_use_dist):    
    m,n,_ = np.shape(X);
    X = X*1.0
    # reshape array such that each row is the RGB components of each pixel
    X_r = np.reshape(X,(m*n,-1))
    
    if is_use_dist:
        locations = np.zeros((m*n,2))
        k = 0
        for i in range(m):
            for j in range(n):
                locations[k] = np.array([i,j])
                k +=1
        X_r = np.concatenate((X_r, locations),axis=1)          
    
    n_centroid = np.shape(centroids)[0]
    distances = np.zeros((m*n, n_centroid))
    for i in range(n_centroid): # iterate through the indicated centroids
        # determine the euclidean distance from each centroid of each pixel 
        if is_use_dist:
            distances[:,i] = ((X_r[:,0]-centroids[i,0])**2 +  #r
                              (X_r[:,1]-centroids[i,1])**2 +  #g
                              (X_r[:,2]-centroids[i,2])**2 +  #b
                              (X_r[:,3]-centroids[i,3])**2 +  #i
                              (X_r[:,4]-centroids[i,4])**2    #j
                             )**0.5
        else:
            distances[:,i] = ((X_r[:,0]-centroids[i,0])**2 + (X_r[:,1]-centroids[i,1])**2  + 
                                 (X_r[:,2]-centroids[i,2])**2)**0.5
    print(type(X_r[0,0]))
    fore_points = (np.argmin(distances,axis=1)<thresh)    
    reshaped_X = np.reshape(fore_points,(m,n))
    
    return reshaped_X

def resize_image(image, process_info, index):
    sampling_ratio = int(process_info[2*index] * 100)
    return imresize(image, sampling_ratio)     

def fel_seg(rgb_img, largeness, sigma, br):
    start = time.time()    
    seg_image = felzenszwalb(rgb_img, scale=largeness, sigma=sigma, min_size=largeness)    
    plt.imshow(seg_image),plt.show()
    fel_seg_ids = (np.unique(seg_image))
    id_segs_dict = {}

    for seg_id in fel_seg_ids:
        id_segs_dict[seg_id] = (seg_id==seg_image)*1        

    fore_mask, back_id = identify_background(seg_image, id_segs_dict, br)
    #print(np.unique(seg_image), back_id)
    fore_mask=fore_mask!=back_id

    cl = 5
    closed_fore_mask = close_image_with_back(fore_mask, cl)

    fore_mask = put_back_boundary(fore_mask, closed_fore_mask, cl)
    fore_mask = morphology.binary_dilation(fore_mask,np.ones((3,3)))

    # choose one object
    fel_labeled_image, fel_object_mask = get_object_mask_from_connected_regions(fore_mask)
        
    print('size ', fel_object_mask.shape, ' Fel processing time ', time.time() - start)    
    return fel_object_mask

def grabcut_seg(rgb_img, mask, num_iter):
    start = time.time()    
    # find the convex hull of the object first
    #cvx_mask = convex_hull_image(mask)        
    #fel_object_mask = (cvx_mask*3).astype(np.uint8)  # 3 in opencv grabcut representing possible fore
    
    fel_object_mask = (mask*3.0).astype(np.uint8)  # 3 in opencv grabcut representing possible fore
    
    
    bgdModel = np.zeros((1,65),np.float64)               # 65 needed by opencv grabcut       
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(rgb_img, fel_object_mask, None, bgdModel, fgdModel, num_iter, cv2.GC_INIT_WITH_MASK)    
    rect_mask = np.where((fel_object_mask==2)|(fel_object_mask==0),0,1).astype('uint8')
    
    se = np.array([[0,1,0],[1,1,1],[0,1,0]])
    closed_fore_mask = morphology.binary_closing(rect_mask==0, se)==0
    pad_boundary(closed_fore_mask, se.shape[0], 0)     # skimage lib after closing sometimes give a black frame

    # get single object
    grab_labeled_image, grab_object_mask = get_object_mask_from_connected_regions(closed_fore_mask.astype(np.uint8),4)
    grab_object_mask = morphology.binary_closing(grab_object_mask, np.ones((2,2)))

    print('size ', grab_object_mask.shape, ' GrabCut processing time ', time.time() - start) 
    return grab_object_mask    

def growcut_seg(rgb_img, rect_mask, safe_zone_len, window_size = 4, num_points=-1):
    start = time.time()
    se = np.ones((safe_zone_len, safe_zone_len))
    grow_fore_points = morphology.binary_erosion(rect_mask, se)
    grow_back_points = morphology.binary_erosion(rect_mask==0, se)
    m,n,_ = rgb_img.shape
    
    if num_points <0:           
        strength = np.zeros((m,n), dtype=np.float64)
        label_img = grow_fore_points*1.0  - grow_back_points*1.0
        strength[np.nonzero(label_img)] = 1.0
        #plt.imshow(label_img), plt.show()
        grow_mask = growcut_cy.growcut((rgb_img), np.dstack((label_img, strength)), window_size=window_size) >0    
    else :        
        #sampling point
        fore_points = get_sample_point(rgb_img, num_points, grow_fore_points)
        back_points = get_sample_point(rgb_img, num_points, grow_back_points)
        label_img =np.zeros((m,n))
        strength = np.zeros((m,n))

        for p in fore_points:
            i,j = p[-2], p[-1]
            label_img[i,j] = 1
            
        for p in back_points:
            i,j = p[-2], p[-1]
            label_img[i,j] = -1    

        strength[np.nonzero(label_img)] = 1.0
        grow_mask = growcut_cy.growcut((rgb_img), np.dstack((label_img, strength)), window_size=window_size) >0
      
    print('size ', grow_mask.shape, ' GrowCut processing time ', time.time() - start)    
    return grow_mask
    
def meanshift_seg(rgb_img, spatial_radius, range_radius, min_density,br):
    start = time.time()
    (segmented_image, labels_image, number_regions) = pms.segment(rgb_img, spatial_radius=spatial_radius,
                                                              range_radius=range_radius, min_density=min_density)
    meanShift_id_segs_dict = {}
    for i in range(number_regions):
        meanShift_id_segs_dict[i] = (i==labels_image)*1        

    fore_mask, back_id = identify_background(labels_image, meanShift_id_segs_dict, br)
    #print(np.unique(labels_image), back_id)
    fore_mask=fore_mask!=back_id

    cl = 5
    closed_fore_mask = close_image_with_back(fore_mask, cl)

    fore_mask = put_back_boundary(fore_mask, closed_fore_mask, cl) 

    labeled_image, object_mask = get_object_mask_from_connected_regions(fore_mask)
    print('size ', object_mask.shape, ' MeanShift processing time ', time.time() - start)    
    return object_mask
    
def randomwalk_seg(rgb_img, mask, safe_zone_len, beta, num_points):
    start = time.time()
    se = np.ones((safe_zone_len, safe_zone_len))
    
    rand_fore_points = morphology.binary_erosion(mask, se)
    rand_back_points = morphology.binary_erosion(mask==0, se)
    
    if num_points <0:
        labels = rand_fore_points*1.0  + 2.0*rand_back_points
        random_walker_mask = random_walker(rgb_img, labels, beta=beta, multichannel=True)==1
        
    else:
        
        #sampling point
        fore_points = segh.get_sample_point(rgb_img, num_points, grow_fore_points)
        back_points = segh.get_sample_point(rgb_img, num_points, grow_back_points)
        labels =np.zeros((m,n))

        for p in fore_points:
            i,j = p[-2], p[-1]
            labels[i,j] = 1    
        for p in back_points:
            i,j = p[-2], p[-1]
            labels[i,j] = 2
           
        random_walker_mask = random_walker(rgb_img, labels, beta=beta, multichannel=True)==1
        
    print('size ', random_walker_mask.shape, ' Random Walk processing time ', time.time() - start)    
    return random_walker_mask
                  
def process_unit(rgb_img, process_info, curr_i, mask):
    unit = process_info[curr_i*2+1]
    operation = unit[0]
    
    if operation is 'Fel':       
        largeness, sigma, br = unit[1], unit[2], unit[3]
        return fel_seg(rgb_img, largeness, sigma, br)
    elif operation is 'GrabCut':
        num_iter = unit[1]
        return grabcut_seg(rgb_img, mask, num_iter)
    elif operation is 'GrowCut':
        safe_zone_len, window_size, num_points = unit[1], unit[2], unit[3]
        return growcut_seg(rgb_img, mask, safe_zone_len, window_size, num_points)
    elif operation is 'MeanShift':
        spatial_radius, range_radius, min_density, br = unit[1], unit[2], unit[3],unit[4]
        return meanshift_seg(rgb_img, spatial_radius, range_radius, min_density, br)
    elif operation is 'RandomWalk':
        safe_zone_len, beta, num_points = unit[1], unit[2], unit[3]        
        return randomwalk_seg(rgb_img, mask, safe_zone_len, beta, num_points)
    else:
        print('parameter format error')

# parameter parser
def seg_image(input_img, process_info):
    start = time.time()
    num_unit = int((len(process_info))/2)
    curr_i = 0
    
    if len(process_info)%2 != 0:
        print('input process info format error')
        return -1
    
    input_m, input_n,_ = np.shape(input_img)
    ori_img,in_u,in_d,in_l,in_r = get_rid_of_black_frame(input_img) #in case there black boundary
    ori_m, ori_n,_ = np.shape(ori_img)
    plt.imshow(ori_img), plt.show()

    rgb_img = np.copy(ori_img)    
    mask = np.zeros((ori_m, ori_n))
    for curr_i in range(num_unit):
        # resize
        rgb_img = resize_image(rgb_img, process_info, curr_i)
        mask = imresize(mask,(rgb_img.shape[0],rgb_img.shape[1]), interp='nearest')> 128
        # processing
        mask = process_unit(rgb_img, process_info, curr_i, mask)   
        # show local result    
        extend = 5
        crop_local_mask, crop_local_image, crop_u, crop_d, crop_l, crop_r = crop_image(mask, rgb_img, extend)
        plt.imshow(mask_rgb(crop_local_mask, crop_local_image)), 
        plt.title('local after previous processing'), plt.show()
        
    if mask.shape[0] != ori_m or mask.shape[1] != ori_n:
        mask = imresize(mask,(ori_m, ori_n), interp='nearest')> 128        
    
    #se = np.array([[0,1,0],[1,1,1],[0,1,0]])
    #mask = morphology.binary_opening(mask,se)
    print('last mask|', 'total processing time ', time.time()-start)
    plt.imshow(mask), plt.show()    
    input_mask = put_back_cropped_mask(in_u,in_d, in_l, in_r, mask, input_m, input_n)
    return input_mask
    
            