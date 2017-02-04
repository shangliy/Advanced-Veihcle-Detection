
import cv2
import glob
import time
import os
import pickle
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from sklearn.model_selection import train_test_split

from util_funcs import *


# Here define the image folders, please change the car_root and noncar_root to your own folder
# The folder structure should be:
#    -car_root
#      - GTI_Far
#      - GTI_Left
#      ...
# Then collecting training data paths into car_paths and noncar_path
car_root = '../project_4_datasets/vehicles/vehicles/'
noncar_root = '../project_4_datasets/non-vehicles/non-vehicles/'
car_folders = os.listdir(car_root)
noncar_folders = os.listdir(noncar_root)

print ("car_folders include:",car_folders)
print ("noncar_folders include:",noncar_folders)

#Collecting image paths 
car_paths = []
for folder in car_folders:
    for image in glob.glob(car_root+folder+"/*"):
        car_paths.append(image)
noncar_paths = []
for folder in noncar_folders:
    for image in glob.glob(noncar_root+folder+"/*"):
        noncar_paths.append(image)

'''
    This function explore the basic information of the training data set;
    Including:
        Car_Images Number;
        Non_Car Images Number;
        Image Shape;
        Image Data Type
'''
Data = data_explore(car_paths,noncar_paths,vio=False)

### Provide parameter for the SVM Training
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 4 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


car_features = extract_features(car_paths, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(noncar_paths, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Stack postive and negative training data together
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

# Save training Model
model = {}
model['svc'] = svc
model['X_scaler'] = X_scaler
model['color_space'] = color_space
model['spatial_size'] = spatial_size
model['hist_bins'] = hist_bins
model['orient'] = orient
model['pix_per_cell'] = pix_per_cell
model['cell_per_block'] = cell_per_block
model['hog_channel'] = hog_channel
model['spatial_feat'] = spatial_feat
model['hist_feat'] = hist_feat
model['hog_feat'] = hog_feat

with open('my_model_LUV.p','wb') as picfile:
    pickle.dump(model, picfile)    
    





