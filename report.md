## Vehicle Detection Project


**Project Goals**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/car_example.png
[image2]: ./report_images/hog_visual.png
[image3]: ./report_images/sliding_windows.png
[image4]: ./report_images/sliding_window.jpg
[image5]: ./report_images/bboxes_and_heat.png
[video1]: ./project_video_out.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the vehicle_detection IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of of color channels as an input to the HOG feature detector. I also trained the linear SVM with various parameter combinations to see what minimized the test error.

#### 3. Training the classifier and selecting HOG and other features.
I trained a linear SVM using:
1. The vehicle/not-vehicle GTI*/extras labeled data was used for training and test purposes.
2. Spatial, color histograms and HOG features were extracted from the labeled data.
3. The data was randomly selected and split 80/20 into training/test sets.
4. Each feature was normalized over the training data and the normalization scaling was applied to the test set and features extracted from the video.
5. Sklearn LinearSVM was used to fit the data and scored 100% on the training and 98.8% on the test - so the classify generalized well to new data and did not show signs of overfitting.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

These are the things I took into consideration when designing the windowing:
1. Windows only covered the road surface and just above the road surface where vehicles might protrude. 
2. Window sizes generally decreased in size the further away from the SDC and toward the center of the road in anticipation of smaller vehicle images.
3. Large amounts of overlapping gave localized heatmaps and allowed efficient high-thresholding.


![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. In my code this can be seen in the last few code cells of the vehicle_detection.ipynb.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

The left most image shows the final bounded boxes superimposed on the lane detection frame, the 2nd image shows the IIR filtered heatmap and the 3rd image is the raw heatmap frame by frame.

![alt text][image5]





---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further:

1. There was an inherent tradeoff between mis-detection and false detection. 
2. Increasing the number of windows increase the detection rate as well as the false detection rate.
3. Raising the threshold filtered out some false detection =s but too high and it reduced detection sensitivity.
4. IIR filtering helped filter the heatmaps to reduce fluctuation.
5. Positioning classification could be used to tailor the sliding window configuration for different lane positions. 



