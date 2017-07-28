# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/model_first.png "Model Visualization First Run"
[image2]: ./examples/model_second.png "Model Visualization Second Run"
[image3]: ./img/recovery1.jpg "Recovery Image"
[image4]: ./img/recovery2.jpg "Recovery Image"
[image5]: ./img/recovery3.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

#### 1. An appropriate model architecture has been employed
My model architecture is based on [Nvidia's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 79-83) 

The data is normalized in the model using a Keras lambda layer (code line 74). 

After the lambda layer the images are cropped so there's no environment and only the road on the images.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 84 & 96). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in the wrong direction to augment the data set.

For details about how I created the training data, see the next section. 

### Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test and fine tune the model. The approach for finding the most suitable model for my problem was to test several architectures and change the layer structure.

My basis for the model architecture is [Nvidia's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

I added one dropout and one maxpool layer after the CNN layers and one dropout between the first two fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My sets were neither overfitting nor underfitting but the problem was that the car still was unable to drive one track autonomously.

I had to change the model several times until I was satisfied with my outcome.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or wasn't driving in the center. To improve the driving behavior I fine tuned my model and tried to gather more data for my training and validation sets.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 YUV image   						| 
| Lambda Layer        	|                         						| 
| Cropping          	| 90x320x3 YUV image       						| 
| Convolution 5x5     	| Strides: 2x2, Output: 43x158x24              	|
| Convolution 5x5     	| Strides: 2x2, Output: 20x77x36               	|
| Convolution 5x5     	| Strides: 2x2, Output: 8x37x48               	|
| Convolution 3x3     	| Output: 3x18x64                              	|
| Convolution 3x3     	| Output: 64x8x31, Dropout rate: 0.5          	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flatten       		| Output: 3968     		                    	|
| Fully connected		| Output: 100, Dropout rate: 0.5     			|
| Fully connected		| Output: 50                          			|
| Fully connected		| Output: 10                          			|
| Fully connected		| Output: 1                          			|

**Here is the visualization of my data:**

![Data Visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps of driving in the right direction on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also recorded one lap of driving in the wrong direction on track one using center lane driving and one lap of driving on track two to make sure there is lots of augmented data.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep on the center of the lane. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

For image preprocessing I converted the images from RGB to YUV color space to reduce the resolution on U and V but to keep Y at full resolution. This helps the CNN to train faster.

To augment the data set I flipped images and angles. I also added a little correction of 0.3 on the steering angle of the left and right camera.

I had 20.370 images without augmentation and a sum of 40.740 images with data augmentation.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 because on the 10th epoch the validation loss was increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
