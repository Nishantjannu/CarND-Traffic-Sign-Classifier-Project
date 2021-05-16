# **Traffic Sign Recognition** 

## Project Writeup

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

[image1]: ./writeup_images/Bar_chart.png "Visualization -1"
[image2]: ./writeup_images/Tabular1.png  "Visualization -2"
[image20]: ./writeup_images/Tabular2.png "Visualization -3"
[image3]: ./writeup_images/3.jpg
[image4]: ./writeup_images/9.jpg
[image5]: ./writeup_images/11.jpg
[image6]: ./writeup_images/13.jpg
[image7]: ./writeup_images/14.jpg
[image8]: ./writeup_images/15.jpg
[image9]: ./writeup_images/22.jpg
[image10]: ./writeup_images/25.jpg
[image11]: ./writeup_images/29.jpg
[image12]: ./writeup_images/37.jpg
[image13]: ./writeup_images/11.jpg
[image14]: ./writeup_images/11_visual.png
[image15]: ./writeup_images/random_pic.jpeg
[image16]: ./writeup_images/Random_visual.png
[image17]: ./writeup_images/Softmax1.png
[image18]: ./writeup_images/Softmax1.png
[image19]: ./writeup_images/Softmax1.png





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Nishantjannu/CarND-Traffic-Sign-Classifier-Project.git)



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* Number of training examples = 34799
* Number of validation examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First is a bar chart showing how the data is distributed across the training set.
It is a plot of the frequency distribution across the labels/signboards.

![alt text][image1]

Below is a tabular representation that compares the distribution of data across training and validation sets.

![alt text][image2]
![alt text][image20]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Out of the many pre-processing techniques available, I chose to use only normalisation. 

I normalised RGB Images (0-255) to (0,1). 

In future iterations, I could improve results using other techniques such as grayscaling, augmentation (rotation, translation, zoom, flips etc) as suggested in the rubric points. Generation of additional data could also be considered to make the distribution of data across training and validation sets more equitable.
As a first step, I decided to convert the images to grayscale because ...


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| 1x1 stride, Valid padding, outputs 28x28x6 : Using a 5x5x3 Filter 	|
| 		Activation		|					RELU								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 2	    | 1x1 stride, Valid padding, outputs 10x10x16 : Using a 5x5x6 Filter  |
| 		Activation		|					RELU								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Fully connected	1	| Input = 400 Output = 120    									|
| 		Activation		|					RELU								|
| 		Regularisation		|					Dropout = 0.5								|
| Fully connected	2	| Input = 120 Output = 84    									|
| 		Activation		|					RELU								|
| 		Regularisation		|					Dropout = 0.5								|
| Fully connected	3 	| Input = 84 Output = 43    									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer (similar to Stochastic Gradient Descent) with a learning rate of 0.001. The Loss function is a summation of the mean cross-entropy with L2 regularisation terms. 

Batch Size: 128, Epochs: 10, Dropout 0.5, L2 Regularisation constant (Lambda) = 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.984
* Validation Accuracy = 0.949
* Testing Accuracy = 0.930

The LeNet-5 model architecture is implemented in this project.

Original validation accuracy: 0.89

Regularisation Techniques were added including Dropout and L2 regularisation -  These were tuned to improve the validation accuracy to above 0.93.

The other hyper-parameters such as learning rate, no.of epochs and batch size were modified as well but the model didn't show significant improvement. As the training as validation accuracy were fairly high (above 0.93) and fairly close (difference of < 0.5)- under-fitting/over-fitting didn't pose a problem.

Other activation functions were also experimented with (ex: sigmoid) but the present ReLU function provided the best results. 

In future iterations, the size of the network could be adjusted and connections altered (using both local and global features in output layer - similar to paper by Yan Le Cun) to further improve accuracy.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]

Bicycle Crossing, Road Work, Yield, Right-of-way at the next intersection and Bumpy Road Signs: All have a triangular outer border. Identification and Classification based on inner features is key.

Speed Limit (60 km/h), No Passing and No vehicles: All have a circular outer border. Identification and Classification based on inner features is key.

The Stop sign is fairly unique. Go straight or left must be distinguished from other directional traffic signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection									|
| Yield					| Yield											|
| Bumpy Road	      		| Bumpy Road					 				|
| Speed limit (60km/h)		|Speed limit (60km/h)      							|
| No vehicles     		| Speed limit (30km/h)   									| 
| Go straight or left     			| Go straight or left 										|
| No passing					| No passing											|
| Road work	      		| Road work				 				|
| Bicycles crossing			| Bicycles crossing     							|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image17] 
![alt text][image18]
![alt text][image19] 

The model faces difficulty on 4 of the 10 images. 
* For the Bumpy road and No passing signs, it is still assigning about 20% probability to the wrong sign out of the two. 
* The Speed Limit (60 kmph) is confused with other speed limits (80,20,30)
* The No Vehicles Sign is inccorectly classified as a Speed Limit sign due to some pixels in the input image being coloured and not completely white.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The features of the network were visualised: For the Conv2 layer 

* On one of the traffic signs: Right-of-way at the next intersection (11)

![alt text][image13] ![alt text][image14]

* On a Random picture unrelated to traffic signs

![alt text][image15] ![alt text][image16]

Clearly, Features related to the sign (such as border, internal animation) are picked up by the model as shown in the first picture. On the second picture, only noise is observed.
