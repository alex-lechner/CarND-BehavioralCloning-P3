# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/model.png "Tarining Model Visualization"
[image2]: ./imgs/recovery_1.jpg "Car Recovery"
[image3]: ./imgs/recovery_2.jpg "Car Recovery"
[image4]: ./imgs/recovery_3.jpg "Car Recovery"
[image5]: ./imgs/recovery_4.jpg "Car Recovery"
[image6]: ./imgs/recovery_5.jpg "Car Recovery"
[image7]: ./imgs/center_driving.jpg "Car Center Driving"
[image8]: ./imgs/driving_one_track.gif "Driving One Track"

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

#### 4. Final Output:

![Data Visualization][image8]

Note: This is gif is accelerated and does not match with the actual speed during this recording. The driving speed was set to 14 mph.

Also look at video.mp4 in this repository for front camera view. The driving speed was set to 12 mph. 

### Model Architecture

#### 1. An appropriate model architecture has been employed
My model architecture is based on [Nvidia's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 74-79) 

The data is normalized in the model using a Keras lambda layer (code line 73). 

After the lambda layer the images are cropped so there's no environment and only the road on the images.

#### 2. Attempts to reduce overfitting in the model

The model contains of four dropout layers in order to reduce overfitting (model.py lines 76, 80, 83 & 85). 

The model was also trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and a learning rate of 0.001 (model.py line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in the wrong direction to augment the data set.

For details about how I created the training data, see the next section. 

### Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test and fine tune the model. The approach for finding the most suitable model for my problem was to test several architectures and change the layer structure accordingly.

My basis for the model architecture is [Nvidia's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

I added one dropout layer after the first two CNN layers, one dropout layer after the last CNN layer and two dropout layers between the first two fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My sets were neither overfitting nor underfitting but the problem was that the car still was unable to drive one track autonomously.

I had to change the model several times until I was satisfied with my outcome.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or wasn't driving in the center. To improve the driving behavior I fine tuned my model and tried to gather more data for my training and validation sets.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-91) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 40x160x3 YUV image   					    	| 
| Lambda Layer        	|                         						| 
| Convolution 5x5     	| Strides: 2x2, Output: 18x78x24              	|
| Convolution 5x5     	| Strides: 2x2, Output: 7x37x36               	|
| Dropout            	| rate: 0.3                                   	|
| Convolution 5x5     	| Strides: 2x2, Output: 2x17x48               	|
| Convolution 3x3     	| Output: 64x15x46                             	|
| Convolution 3x3     	| Output: 62x13x64                            	|
| Dropout            	| rate: 0.3                                   	|
| Flatten       		| Output: 3968     		                    	|
| Fully connected		| Output: 100                        			|
| Dropout            	| rate: 0.3                                   	|
| Fully connected		| Output: 50                          			|
| Dropout            	| rate: 0.3                                   	|
| Fully connected		| Output: 10                          			|
| Fully connected		| Output: 1                          			|

Total params: 5,258,555

Trainable params: 5,258,555

Non-trainable params: 0

And here is a visualization of the training process:

![Data Visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps of driving in the right direction on track one using center lane driving. Here is an example image of center lane driving:

![Car Center Driving][image7]

I also recorded one lap of driving in the wrong direction on track one and one lap of driving on track two to make sure there is lots of augmented data.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep on the center of the lane. These images show what a recovery looks like:

![Car Recovery][image2] ![Car Recovery][image3] ![Car Recovery][image4] ![Car Recovery][image5] ![Car Recovery][image5] ![Car Recovery][image6]

For image preprocessing I resized the images from 320x160 pixel to 160x80 pixel in order to reduce image size and to make the network train faster. After that I cropped the images to 160x40 pixel so I only had the important parts of the images for my CNN and then I converted the images from RGB to YUV color space to reduce the resolution on U and V but to keep Y at full resolution. This helped the CNN to train faster.

**Important note:** I also added this image preprocessing technique in drive.py (lines 83-92) before the images are fed to the prediction model. The resizing and cropping process is necessary because otherwise keras throws an error and would not execute. The conversion from RGB to YUV space is the most important part because if the model is trained in YUV color space but the prediction model receives RGB-images the car would not drive around one track safely and autonomously.

Although I implemented data augmentation in my code (model.py 55-58) I didn't use it in my generator because the training process would have taken too long. For the left and right camera images I added a little correction of 0.3 on the steering angle.

In sum I had 30.702 images without augmentation to train my model.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. For training I used 10 epochs and a batch size of 64. Although 10 epochs may be too much and not necessary for my data set I implemented an early stopping callback that would stop the training process if the validation loss would not decrease after two consecutive epochs.
