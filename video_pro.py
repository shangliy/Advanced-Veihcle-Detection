import os
import sys
import argparse
import cv2
import math 
import glob
import time
import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.misc import imshow,imsave
from sklearn.model_selection import train_test_split
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import watershed, disk
from skimage.filters import rank
from scipy import ndimage as ndi
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from util_funcs import *

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
frame_dec = Frame()
fram_count = 0
fram_sta = 150
check_sta = 1300

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


img_vis = True

def process_image(image):
    global fram_count
    

    img_shape = image.shape
    draw_image = np.copy(image)
    image_todraw = np.copy(image)
    
    car_detect = 0
    pre_box = []

    #initilize the frame_dec 
    if (fram_count  == 0):
        frame_dec.heat_img = np.zeros((img_shape[0],img_shape[1]))
    
    # Transfer image to float
    image = image.astype(np.float32)/255.

    
    find_car = False

    if (frame_dec.car_num == 0 ):
        find_car==False
    else:
        find_car = True
    #Skip some pre frames
    if (fram_count >= fram_sta):
        img_vis = False
    else:
        fram_count += 1
        return image*255.

    # windows size, search area and overlap parameter
    window_scale_w = [64,80,128,160,213,320,350]
    window_scale_h = [64,80,120,100,120,120,120]
    x_start = [640,640,640,750,854,960,960]
    x_stop = [None,None,None,None,None,None,None,None]
    y_start_stop = [[400, 600],[400, 600],[400, 650],[400, 650],[400, 650],[400, 650],[400, 650]]
    overlap = [0.5,0.5,0.5,0.5,0.6,0.6,0.7]
    
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
           
    draw_windws(draw_image, hot_windows,img_vis,fram_count,check_sta)
    heat_img = heat_generation(img_shape,hot_windows,frame_dec,fram_count)    
    labels = label(heat_img)

    if ( labels[1]==0 ):
        print (fram_count)
        imsave(str(fram_count)+".jpg",image)
            

    frame_dec.heat_img = heat_img  
    frame_dec.heat_img[[heat_img >0]] = 1
            
    car_detect = labels[1]
    if (img_vis):
        plt.imshow(labels[0], cmap='gray')
        plt.show()
            
            
    if (labels[1]>0):
        image_todraw,fr_boxes,car_detect = draw_labeled_bboxes(image_todraw , labels,frame_dec.boxes,check=False)
        pre_box.append(fr_boxes)
    else:
        image_todraw = draw_labeled_bboxes(image_todraw , labels)
            
    frame_dec.car_num = car_detect
    frame_dec.boxes = pre_box
    
    
    if img_vis:
        plt.imshow(labels[0], cmap='gray')
        plt.show()
    
    
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
