# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./relevant_images/training_distribution.png "Training Visualization"
[image2]: ./relevant_images/validation_distribution.png "Validation Visualization"
[image3]: ./relevant_images/testing_distribution.png "Testing Visualization"
[image4]: ./test_images/14.jpg width="200" height="400" "Traffic Sign 5"
[image5]: ./test_images/11.jpg width="200" height="400" "Traffic Sign 4"
[image6]: ./test_images/12.jpg width="200" height="400" "Traffic Sign 1"
[image7]: ./test_images/15.jpg width="200" height="400" "Traffic Sign 2"
[image8]: ./test_images/18.jpg width="200" height="400" "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data labels are distributed in the entire data

![alt text][image1] ![alt text][image2] ![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


I normalized the image data to speeds up learning and achieve faster convergence.

All images were normalized to values between [-1 1]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout					|		probability = 0.8										|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout					|		probability = 0.8										|
| Flatten     	| outputs 400 	|
| Fully connected		| Input = 400, Output = 120       						|
| RELU					|												|
| Dropout					|		probability = 0.8										|
| Fully connected		| Input = 120, Output = 84       						|
| RELU					|												|
| Dropout					|		probability = 0.8										|
| Fully connected		| Input = 84, Output = 43       						|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an a batch size of 128. The model was trained for 80 epochs and the loss was optimized with an Adam optimizer. Finally a learning rate of 0.001 is chosen. The final was model is saved as 'lenet_final'

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.951 
* test set accuracy of 0.936

In the beginning, different approaches were chosen in the hopes of getting the accuracy up. My initial architecture had no dropouts. This only got my accuracy up to around 80%. I then tried to play around with the batch size and learning rate, however, as I increased the learning rate the model was not learning almost anything. A lower learning rate just did not achieve a higher percentage either. So I decided to include dropout layers inbetween each layer in the architecture. It worked! I started to see accuracies around 90%. To get those last extra percentages, I just kept playing around with the learning rate and the number of epochs until i reached the final validation accuracy of 95.1%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection										|
| Stop					| Stop											|
| No vehicles	      		| No vehicles				 				|
| General caution			| General caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.6%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all chosen new test images, the model is very sure of his classifications with an accuracy of 99% for all of them, except of the general caution image which got an accuracy of 94%.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road   									| 
| .99     				| Right-of-way at the next intersection 										|
| .99					| Stop											|
| .99	      			| No vehicles					 				|
| .94				    | General caution      							|
