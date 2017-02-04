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


[imagecol1]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_RGB_0.png?raw=true
[imagecol2]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_RGB_1.png?raw=true
[imagecol3]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_RGB_2.png?raw=true
[imagecol4]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HSV_0.png?raw=true
[imagecol5]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HSV_1.png?raw=true
[imagecol6]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HSV_2.png?raw=true
[imagecol7]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_LUV_0.png?raw=true
[imagecol8]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_LUV_1.png?raw=true
[imagecol9]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_LUV_2.png?raw=true
[imagecol10]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HLS_0.png?raw=true
[imagecol11]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HLS_1.png?raw=true
[imagecol12]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_HLS_2.png?raw=true
[imagecol13]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YUV_0.png?raw=true
[imagecol14]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YUV_1.png?raw=true
[imagecol15]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YUV_2.png?raw=true
[imagecol16]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YCrCb_0.png?raw=true
[imagecol17]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YCrCb_1.png?raw=true
[imagecol18]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_YCrCb_2.png?raw=true


[imagehog1]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_rgb.png?raw=true
[imagehog2]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_hsv.png?raw=true
[imagehog3]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_LUV.png?raw=true
[imagehog4]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_hls.png?raw=true
[imagehog5]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_yuv.png?raw=true
[imagehog6]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1_hog_ycc.png?raw=true

[imagehog7]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_rgb.png?raw=true
[imagehog8]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_hsv.png?raw=true
[imagehog9]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_LUV.png?raw=true
[imagehog10]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_hls.png?raw=true
[imagehog11]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_YUV.png?raw=true
[imagehog12]: https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2_hog_ycc.png?raw=true



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

This is the README that includes all the rubric points and how you addressed each one.
![whole](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/Svm_Training_whole.png?raw=true)

###Histogram of Oriented Gradients (HOG)

####1. Explain how extracted HOG features from the training images.

The pipeline of the feature extraction is shown in the figure.
The detail Codes are in the train_model.py and util_funcs.py

![hog](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/Svm_Training_hog.png?raw=true)

Step 1. Work start with Reading all the `vehicle` and `non-vehicle` images.
* This work finised in line **21 to 44** in **`train_model.py`**. Got the image folders and collected all image paths;
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

| Channels  |      RGB      | HSV | LUV|HLS|YUV|YCrCb|
|---------- |:-------------:|------:|------:|------:|------:|------:|
| channel_1 |  ![alt text][imagecol1] |  ![alt text][imagecol4] | ![alt text][imagecol7]| ![alt text][imagecol10] |  ![alt text][imagecol13] |  ![alt text][imagecol16] |
| channel_2 |  ![alt text][imagecol2] |  ![alt text][imagecol5] | ![alt text][imagecol8]| ![alt text][imagecol11] | ![alt text][imagecol14] | ![alt text][imagecol17] |
| channel_3 |  ![alt text][imagecol3] |  ![alt text][imagecol6] | ![alt text][imagecol9]| ![alt text][imagecol12] | ![alt text][imagecol15] | ![alt text][imagecol18] |
 
 
 * Step 3.3: Hog Feature extraction using **`skimage.hog()`**, from line **81 to 98**  in **`util_funcs.py`**;

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:



| Color_Space  |          |
|---------- |:-------------:|
| Original Image  |     ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true)      |
| RGB  |      ![alt text][imagehog1]      |
| HSV  |      ![alt text][imagehog2]      |
| LUV  |      ![alt text][imagehog3]      |
| HLS  |      ![alt text][imagehog4]      |
| YUV  |      ![alt text][imagehog5]      |
| YCrCb  |     ![alt text][imagehog6]      |



 
 * Step 3.4: binned color features and color histogram features  extraction, from line **101 to 117**  in **`util_funcs.py`**.;
 
 * Step 3.5: Combined all features together


####2. Explain how you settled on your final choice of HOG parameters.

The HOG parameters include **`color_space`** ,**`orient`**, **`pix_per_cell`**, **`cell_per_block`**, and **`hog_channel`**

2.1 **`color_space`**
 
 Also, we can compare the hog features between  the `vehicle` and `non-vehicle` images
 
| Original Image  |      **RGB**      |  
|---------- |:-------------:|------:|
| `vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) |  ![alt text][imagehog1] | 
| `non-vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true) |  ![alt text][imagehog7] | 
| Original Image  |      **HSV**      | 
| `vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) | ![alt text][imagehog2] | 
| `non-vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true) |  ![alt text][imagehog8] |
| Original Image  |      **HLS**      | 
| `vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) | ![alt text][imagehog4] | 
| `non-vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true) |  ![alt text][imagehog10] |
| Original Image  |      **YUV**      | 
| `vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) | ![alt text][imagehog5] | 
| `non-vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true) |  ![alt text][imagehog11] |
| Original Image  |      **YCrCb**      | 
| `vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/car_1.jpg?raw=true) | ![alt text][imagehog6] | 
| `non-vehicle`![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/noncar_2.jpg?raw=true) |  ![alt text][imagehog12] |

From the comparision above, we can see for
>* RGB: The Hog features of three channels are almost same, and and thus not show enough information compared to other color space using three channels;
>* HSV/HLS: Both features are good,but we they do kind of worse than the YCbCr, detail explained below;
>* YUV/YCrCb: 
>>*	From the hog images, we can see the hog features of YUV/YCrCb can detect the shape of the car;
>>*	The difference between vehicle and non-vehicle are large enough to make decision
>>*    Compared to HSV/HLS, the features strength are stronger and more robust, and the difference are more distinguishable;
>>*    These two are basically same, we choose YCrCb for general;

Thus, I choose to use **YCrCb** color space;

2.2 **`orient`**
According the guidence, the orient in general from 8 to 13; thus I tune the parameter based on test performance;

| orient  |    accuracy %     |
|---------- |:-------------:|
| 8  |     97.6 |
| 9 |      98.5      |
| 10  |      98.2      |
| 11  |    98.7      |
| 12  |      98.6      |
| 13  |      98.5    |

We can see the the result is good enough when orient equal to 9, considering the calculation speed, I choose orient = 9

2.3 **`pix_per_cell`** and **`cell_per_block`**

The image size = 64

| pix_per_cell  |     cell_per_block     |  Hog Feature length    | accuracy %     |
|---------- |:-------------:|:-------------:|:-------------:|
| 4 |      2      | 24300 | 98.7 |
| 8 |      2      | 5292 | 98.5 |
| 8  |    4      | 10800 | 97.6 |
| 16  |      2      | 972 | 97.5 |
| 16  |      4    | 432 | 96.3 |


Considering the feature vectore length and accuracy, I choose **pix_per_cell = 8, cell_per_block=2**

2.4 **`hog_channel`**
Cause I choose **YCrCb** color space, thus I use **hog_channel=='ALL'** for more valuable information;

####3. Describe how you trained a classifier using your selected HOG features (and color features if you used them).

The pipline of training classifier is shown below:
![svm](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/Svm_Training_svm.png?raw=true)
>* Step 1.Normaliztion features using `StandardScaler()` considering the different value range of features, line **82 to 87**  in **`train_model.py`**;
* Step 2.Split the all training data into train data and test data to weak the overfitting using `sklearn.model_selection.train_test_split` with `split_ratio = 0.1`,line **94**  in **`train_model.py`**;	
* Step 3.Train the SVM classifier `svc = LinearSVC()` using `svc.fit(X_train, y_train)` with training data,line **100 to 105**  in **`train_model.py`**;
* Step 4.Test the classifier performace using `svc.score(X_test, y_test)` with test data,line **108**  in **`train_model.py`**;
* Step 5.Save the trained model to pickle file for later usage,line **113 to 125**  in **`train_model.py`**.


###Sliding Window Search

####1. Describe how implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The key ideas of sliding window search are to capture the valuable information as more as possible while using as smaller as numbers of windows used;
The implemention is to use function **`slide_window()`** from **line 122 to 156** in **`util_funcs.py`** and set parameters in **`video_pro.py`** from line **103 to 109**.
The function **search within in the set srea with windows of different sizes**;

First,  I observe the video to get the rough idea of size of car, the size should be from [64,64] to [350,150];
| samll car  |    big car    |
|---------- |:-------------:|
| ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/small.png?raw=true)  |     ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/big.png?raw=true) |

Secondly, to remove unnecessary information to increase the speed the accuracy,like the sky, left area of the car(cause the car running the left most), I limited the searching area, implemented in **`video_pro.py`** from line **106 to 109**

> x_start = [640,640,640,750,854,960,960]
    x_stop = [None,None,None,None,None,None,None,None]
    y_start_stop = [[400, 600],[400, 600],[400, 650],[400, 650],[400, 650],[400, 650],[400, 650]]
 ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/area.png?raw=true)
  
Thirdly, to avoid the border missing, I need to choose windows size that could be divide by the search area,implemented in **`video_pro.py`** from line **103 to 106**
> For example, if x_start = 640, x_stop = 1280, the width = 640, then the width%window_size ==0 
 ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/border_effect.png?raw=true)
Besides, I choose different size for width and height, Thus, I Choose:
>window_scale_w = [64,80,128,150,213,320,350]
window_scale_h = [64,80,120,120,120,120,150]

Finally, to speed up the search speed as well as to avoid big jump from window to window, I set differnt overlaps for different window size while keep the overlap as small as possible;
>overlap = [0.5,0.5,0.5,0.5,0.6,0.6,0.7]

So, the slide windows are shown in below:
 ![](https://github.com/shangliy/Advanced-Veihcle-Detection/blob/master/reference_imgs/slide_windows.jpg?raw=true)


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

