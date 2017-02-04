import os
import sys
import argparse

import numpy as np
import cv2
import math 

import pickle

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
from lesson_functions import *
from scipy.misc import imshow,imsave
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import watershed, disk
from skimage.filters import rank
from scipy import ndimage as ndi
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label


# Define a class to receive the characteristics of each line detection
class Frame():
    def __init__(self):
        
        # x values of the last n fits of the line
        self.heat_img = [] 
        #average x values of the fitted line over the last n iterations
        self.car_num = 0
        #
        self.boxes = []
        self.preleft = 1080
        self.pretop = 720



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', dest='video_input', type=str,\
                        help='The path of input video')
parser.add_argument('-o', dest='video_output', type=str,\
                        help='The path of output video')
args = parser.parse_args()
print(args)

video_input = args.video_input
video_output = args.video_output
frame = 0 
frame_cut = 10

with open('my_model.p','rb') as picfile:
        model = pickle.load(picfile)  
img_vis = False

frame_dec = Frame()
fram_count = 0
fram_sta = 145
check_sta = 1300
def process_image(image):
    global fram_count
    

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

    #y_start_stop = [400, 600] # Min and max in y to search in slide_window()

    img_shape = image.shape
    
    car_detect = 0
    pre_box = []
    if (fram_count  == 0):
        frame_dec.heat_img = np.zeros((img_shape[0],img_shape[1]))
    
    draw_image = np.copy(image)
    new_img = np.copy(image)
    image_todraw = np.copy(image)
    
    image_back = np.copy(image)
    image = image.astype(np.float32)/255.

    
    
    #print(y_start_stop[0])
    windows_result  = []
    hot_windows = []

    start_win = 0
    find_car = False

    if (frame_dec.car_num == 0 ):
        find_car==False
    if (frame_dec.car_num == 0 and find_car==False):
        
        img_vis = False
    else:
        img_vis = False
        #img_vis = True
        find_car = True
    
    
    if (fram_count >= fram_sta):
        img_vis = False
    else:
        fram_count += 1
        return image*255.

    print(find_car)

    window_scale_w = [64,80,128,150,160,213,320,350]
    window_scale_h = [64,80,120,120,100,120,120,120]
    x_start = [640,640,640,750,640,640,960,930]
    x_stop = [None,None,None,None,None,None,None,None]
    y_start_stop = [[400, 600],[400, 600],[400, 650],[400, 650],[400, 650],[400, 650],[400, 650],[400, 650]]
    overlap = [0.7,0.7,0.5,0.5,0.5,0.5,0.5,0.5]
    for i in range(start_win,len(window_scale_w)):
            sta_name = fram_count*7+i
            win_size_w = window_scale_w[i]
            win_size_h = window_scale_h[i]
            
            windows = slide_window(image, x_start_stop=[x_start[i], x_stop[i]], y_start_stop=y_start_stop[i], 
                        xy_window=(win_size_w, win_size_h), xy_overlap=(overlap[i], overlap[i]))
            hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat, previous_windows=hot_windows,sta_name=sta_name) 
           
    draw_windws(draw_image, hot_windows,img_vis,fram_count,check_sta)
    heat_img = heat_generation(img_shape,hot_windows,frame_dec,fram_count)
                
    labels = label(heat_img)
    #print(len(hot_windows))
    if ( labels[1]==0 ):
        print (fram_count)
        imsave(str(fram_count)+".jpg",image)

    #print (labels[1])

    #print(frame_dec.car_num) 
            
    ori_heat_img = heat_img
    frame_dec.heat_img = ori_heat_img 
    ori_heat_img[ori_heat_img >0] = 1
    frame_dec.heat_img[[ori_heat_img >0]] = 1
            
    car_detect = labels[1]
    if (fram_count>check_sta):
        plt.imshow(labels[0], cmap='gray')
        plt.show()
            
            
    if (labels[1]>0):
        image_todraw,fr_boxes,car_detect = draw_labeled_bboxes(image_todraw , labels,frame_dec.boxes,check=False)
        pre_box.append(fr_boxes)
    else:
        image_todraw = draw_labeled_bboxes(image_todraw , labels)
                    
    
    
    frame_dec.car_num = car_detect
    frame_dec.boxes = pre_box
    
    '''
    if img_vis:
        plt.imshow(labels[0], cmap='gray')
        plt.show()
    '''
    
    # Draw bounding boxes on a copy of the image
    
    
    # Display the image
    if img_vis:
        plt.imshow(image_todraw)
        plt.show()
    if (fram_count>check_sta):
        plt.imshow(image_todraw)
        plt.show()
    fram_count += 1
    return image_todraw

    
clip1 = VideoFileClip(video_input)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output, audio=False)
