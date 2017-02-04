import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imshow,imsave


def save_all_ch(name,img):
    for i in range(3):
        imsave(name+'_'+str(i)+'.png',img[:,:,i])

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


# Read a color image
img = cv2.imread('reference_imgs/noncar_3.jpg')
#save_all_ch('car_1_RGB',img)

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
# Plot and show
plot3d(img_small_RGB, img_small_rgb)
plt.show()


'''
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
save_all_ch('car_1_HSV',img_small_HSV)
plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()

img_small_LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
save_all_ch('car_1_LUV',img_small_LUV)
plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
plt.show()   

img_small_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
save_all_ch('car_1_HLS',img_small_HLS)
plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
plt.show()     

img_small_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
save_all_ch('car_1_YUV',img_small_YUV)
plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
plt.show()      

img_small_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
save_all_ch('car_1_YCrCb',img_small_YCrCb)
plot3d(img_small_YCrCb, img_small_rgb, axis_labels=list("YCrCb"))
plt.show()
'''
img_float =  img.astype(np.float32)/255.
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    
    features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
    return features, hog_image
#feature_image = img_float
feature_image = cv2.cvtColor(img_float, cv2.COLOR_BGR2LUV)
hog_features = []
num = 0

orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block 

for channel in range(feature_image.shape[2]):
    num += 1
    _,hog_img = get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)
    plt.subplot(1, 3, num)
    plt.imshow(hog_img,cmap='gray')

plt.show()

