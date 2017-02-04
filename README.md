##Advanced Veihcle Detection
###This is the README that includes all the rubric points and how you addressed each one.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/img50.jpg
[image6]: ./examples/example_output.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

This is the README that includes all the rubric points and how you addressed each one.

###Histogram of Oriented Gradients (HOG)

####1. Explain how extracted HOG features from the training images.

The pipeline of the feature extraction is shown in the figure.
The detail Codes are in the train_model.py and util_funcs.py

Step 1. Work start with Reading all the `vehicle` and `non-vehicle` images.
* This work finised in line **21 to 44** in **train_model.py**. Got the image folders and collected all image paths;
Step 2. Explore the Data Sep Properties;
* This work used function **`data_explore()`** defined in line **17 to 48** in **util_funcs.py**.

>The number of noncar images is  10040
>The shape of  images is  (64, 64, 3)
>The data type of image is float32

Sample images:

| vehicl   |      non-vehicle    |
|----------|:-------------:|
| ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) |  ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_1.jpg?raw=true) |
| ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_3.png?raw=true)|    ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true)   | 
| ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_4.png?raw=true) | ![alt text](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_4.jpg?raw=true)|  

Step 3.Extract features for images of `vehicle` and `non-vehicle`. 
* Using Function **`extract_features()`** defined in line **53 to 77** in **`util_funcs.py`**.
* Detail implemented in function **`single_img_features()`** defined in line **201 to 253** in **`util_funcs.py`**.,includes:
 * Step 3.1: Reading image using mpimg.imread(), line **63** in **`util_funcs.py`**;
 * Step 3.2: Colore Space Transform using cv2.cvtColor(), Transform image from RGB to desired Color Space from line **209 to 221**  in **`util_funcs.py`**;

[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_not_car.png
[image3]: ./examples/car_not_car.png
[image4]: ./examples/car_not_car.png
[image5]: ./examples/car_not_car.png
[image6]: ./examples/car_not_car.png
[image7]: ./examples/car_not_car.png
[image8]: ./examples/car_not_car.png
[image9]: ./examples/car_not_car.png
[image10]: ./examples/car_not_car.png
[image11]: ./examples/car_not_car.png
[image12]: ./examples/car_not_car.png
[image13]: ./examples/car_not_car.png
[image14]: ./examples/car_not_car.png
[image11]: ./examples/car_not_car.png
[image12]: ./examples/car_not_car.png
[image13]: ./examples/car_not_car.png
[image14]: ./examples/car_not_car.png

 	| Channels  |      RGB      | HSV | LUV|HLS|YUV|YCrCb|
    |----------|:-------------:|------:||------:||------:||------:||------:|
    | channel_1 |  left-aligned | $1600 |$1600 |$1600 |$1600 |$1600 |
    | channel_2 |    centered   |   $12 |$1600 |$1600 |$1600 |$1600 |
    | channel_3 | right-aligned |    $1 |$1600 |$1600 |$1600 |$1600 |
 
 * Step 3.3: Hog Feature extraction using **`skimage.hog()`**, from line **81 to 98**  in **`util_funcs.py`**;
 * Step 3.4: binned color features and color histogram features  extraction, from line **101 to 117**  in **`util_funcs.py`**.;
 * Step 3.5: Combined all features together

* Color Space Exploration:
| Original Image  |      RGB      | HSV | LUV|HLS|YUV
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used blob detection in Sci-kit Image (Determinant of a Hessian [`skimage.feature.blob_doh()`](http://scikit-image.org/docs/dev/auto_examples/plot_blob.html) worked best for me) to identify individual blobs in the heatmap and then determined the extent of each blob using [`skimage.morphology.watershed()`](http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html). I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap and bounding boxes overlaid on a frame of video:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

