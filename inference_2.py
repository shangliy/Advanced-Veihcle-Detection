import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from util_funcs import *
from scipy.misc import imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import watershed, disk
from skimage.filters import rank
from scipy import ndimage as ndi
from scipy.ndimage.measurements import label

with open('my_model.p','rb') as picfile:
        model = pickle.load(picfile)

svc = model['svc'] 
X_scaler = model['X_scaler']  
color_space = model['color_space']  
spatial_size = model['spatial_size']  
hist_bins = model['hist_bins']  
orient = model['orient']  
pix_per_cell = model['pix_per_cell']  
cell_per_block = model['cell_per_block']  
hog_channel = model['hog_channel']  
spatial_feat = model['spatial_feat']  
hist_feat = model['hist_feat']  
hog_feat = model['hog_feat']  

def frame_pro(image,img_vis):
       
    img_shape = image.shape
    
    draw_image = np.copy(image)
    new_img = np.copy(image)

    image = image.astype(np.float32)/255.

    window_scale_w = [64,80,128,160,213,320,350]
    window_scale_h = [64,80,120,100,120,120,120]
    x_start = [640,640,640,750,854,960,960]
    x_stop = [None,None,None,None,None,None,None,None]
    y_start_stop = [[400, 600],[400, 600],[400, 650],[400, 650],[400, 650],[400, 650],[400, 650]]
    overlap = [0.5,0.5,0.5,0.5,0.6,0.6,0.7]
    windows_result  = []
    
    hot_windows = []
    for i in range(len(window_scale_w)):
        win_size_w = window_scale_w[i]
        win_size_h = window_scale_h[i]

        windows = slide_window(image, x_start_stop=[x_start[i], x_stop[i]], y_start_stop=y_start_stop[i], 
                        xy_window=(win_size_w, win_size_h), xy_overlap=(overlap[i], overlap[i]))
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, previous_windows=hot_windows)
    
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6) 
    plt.imshow( window_img)
    plt.show()
    heat_img = np.zeros((img_shape[0],img_shape[1]))
    for box in hot_windows:
        heat_img[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    plt.imshow(heat_img)
    plt.show()

    heat_img = apply_threshold(heat_img,5)
    plt.imshow(heat_img)
    plt.show()
    
    labels = label(heat_img)
    print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')
    plt.show()

    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
        # Return the image
        return img


    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image)*255, labels)
    # Display the image
    plt.imshow(draw_img/255.)
    plt.show()

image = mpimg.imread('test_images/test6.jpg')

frame_pro(image,img_vis=True)