# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteupImages/DatasetCount_BarGraph.JPG "Bar Graph for DataSets"
[image2]: ./WriteupImages/DatasetCount_BarGraph.JPG "Random Training Datasets"
[image3]: ./WriteupImages/AugmentedImages.jpg "Original Plus Augmented Data"
[image4]: ./WriteupImages/NewImagesWithLabels.JPG "New Traffic Signs"
[image5]: ./WriteupImages/Model_Learning_Curve_Loss.JPG "Learning Curve and Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 Images
* The size of the validation set is 4410 Images
* The size of test set is 12630 Images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

![alt text][image2]

#### 2. Include an exploratory visualization of the dataset.

For Visualizing the type of datasets we have, I used Bar Graphs for each kind of data set and found some classes have more images than others.
Please find the Bar Graph in attached image below:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Preprocessing the data includes Augmentation and Normalization techniques as described below:

We can observe that images from the same class can be represented quite differently in the dataset. Generally, there can be different lighting conditions, image can be blurred, rotated, scaled or translated. Indeed, these are samples which are extracted from real world images. And our model have to handle all of these conditions. So, it’s probably better not to truncate our dataset in order to obtain data balance. So for making our model more robust, I included following Augmentation techniques:

* Image Scaling : Zooming of 12.5 %
* Image Translation : In each direction for about 3 Pixels
    1. North Direction
    2. South Direction
    3. East Direction
    4. West Direction
    5. North-East Direction
    6. South-East Direction
    7. North-West Direction
    8. South-West Direction
* Image Rotation : 5 Degrees Rotation
    1. Clockwise Rotation
    2. Counter-Clockwise Rotation
    
 
As a last step, I normalized the image data in the range of -1 to 0.99 because it prevents from the numerical unstabilities which can occur when the data resides far away from zero.
**Normalised Images Pixels = (InputImagePixelValues -128.0)/128.0**

I decided to generate additional data because it would make the model robust.

Here is an example of an original image and all the augmented images. First Image is the original Image from Dataset:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        								| 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   										| 
| Convolution 5x5     	| 32 Filters,1x1 stride, VALID padding, outputs 28x28x32 	|
| RELU					|															|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 							|
| Convolution 3x3	    | 64 Filters,1x1 stride, VALID padding, outputs 12x12x64	|
| RELU					|															|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64   							|
| Convolution 1x1	    | 128 Filters,1x1 stride, VALID padding, outputs 6x16x128   |
| RELU					|															|
| Max pooling	      	| 2x2 stride,  outputs 3x3x128   							|
| Dropout Layer 		| 50 % Dropout 												|
| Flatten       		| 3x3x128,  Outputs 1152									|
| Fully connected		| 1152 ,  Outputs 860 										|
| Fully connected		| 860  ,  Outputs 344										|
| Fully connected		| 344  ,  Outputs  43										|
| Softmax				| Final Output of the Model									|
|						|															|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following HyperParameters:
* Learning Rate of 0.0004
* Batch Size of 640
* Number of Epochs used were 15
* Dropout percentage was 60%
* Optimizer Used was *AdamOptimizer*  

Batch was increased becasue of increase in training dataset by 80%.
Because of Slow learning Rate and Dropouts, I had to use Epoch of 15.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 99.9 %
* Validation set accuracy of 94.5 %
* Test set accuracy of 94.6 % 

Below, there are details of intermediate steps that I took and the corresponding validation accuracies after training.
Initial LeNet model, choosing input images color representation — 90 % and an Underfitting Model
Input images normalization — Almost same as before
After Training set augmantation — 93 % but still Underfitting Model
Increase in Kernels and Hyperparameter optimizations — 97.3 % but Overfitting Model
Introduce Dropouts - 93 % Not a Balanced Model
Added 1 x 1 Convolution Layer (LeNet 5,5,1) - 94 % And Not Balanced Model and New Images Prediction Accuracy 60% 
Changed Convolution size from 5-->3 in 2nd Layer (LeNet 5,3) - 94 % And Balanced Model But New Images Prediction Accuracy 80% 
Added 1 x 1 Convolution Layer (LeNet 5,3,1)- 94.6 % And Balanced Model with New Images Prediction Accuracy 93.8% 

Below is the Learning Curve and the Losses Depiction:
![alt text][image5] 

LeNet Model was chosen as Base Model. It has proven to be succesful in predicting Traffic Signs. But with the change of backgrounds and clearity of images, its accuracy goes low.
Hence modifications and diverse datasets are required for model to be more efficient in classifying Traffic Signs.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

For checking model performance more efficiently. I have chosen 16 new images from Internet. Here are five German traffic signs that I found on the web:

![alt text][image4] 

The most difficult images and reasons for being difficult to predict are : 
4th Image (Go Straight or Left) : Very Blurry and mixed with background
9th Image (Raod Work) : Almost Unclear to human eye 
13th Image(70 Kph Speed Limit): Sky as Background and similarity with 20 kph sign
14th Image(Priority Road) : Different Background, Pretty much mixed with it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	       					| 
|:---------------------:|:-----------------------------------------:| 
| Keep Right     		|  Keep Right     							| 
| Gen Caution  			|  Gen Caution  							|
| No Entry				|  No Entry									|
| Go Straight or Left	|  Go Straight or Left		 				|
| 80 kph Limit			|  80 kph Limit								|
| Bumpy Road     		|  Bumpy Road     							| 
| Take right Ahead		|  Take right Ahead							|
| Yield 				|  Yield 									|
| Road Work         	|  Road Work         		 				|
| Stop Sign 			|  Stop Sign 								|
| 30 Kph Limit     		|  30 Kph Limit    							| 
| Ahead Only  			|  Ahead Only  								|
| 70 kph Limit			|  `20 kph Limit`							|
| Priority Road     	|  Priority Road     		 				|
| Roundabout			|  Roundabout								|
| 50 kph Limit			|  50 kph Limit								|

The model was able to correctly guess 15 of the 16 traffic signs, which gives an accuracy of 93.8%. This compares favorably to the accuracy on the test set of 94.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

Except for the Bumpy Road Sign and the 70kph Limit sign, model was pretty much sure of the predictions.The top soft max probabilities were:

| Probability         	|     Prediction							| 
|:---------------------:|:-----------------------------------------:| 
| 1.0					| Keep Right     							| 
| .99					| Gen Caution   							|
| 1.0					| No Entry									|
| .99	      			| Go Straight or Left		 				|
| .95				    | 80 kph Limit								|
| .49					| Bumpy Road     							| 
| 1.0					| Take right Ahead							|
| 1.0					| Yield 									|
| .99					| Road Work         		 				|
| 1.0					| Stop Sign 								|
| 1.0					| 30 Kph Limit    							| 
| 1.0					| Ahead Only  								|
| .91					| 20 kph Limit 								|
| 1.0					| Priority Road     		 				|
| .99					| Roundabout								|
| 1.0					| 50 kph Limit								|

Although for our wrongly predicted Image of 70 kph Limit, the second most likely prediction was indeed 70kph.
