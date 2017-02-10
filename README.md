# webCamEmocognizer
A cool emotion detector using your laptop/desktop webcam.

The data for this work is taken from the kaggle competiotion: 
Challenges in Representation Learning: Facial Expression Recognition Challenge https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

The dataset provided in the competion consists of gray scale images which are 48 x 48 in dimension and the corresponding labels consisting of 7 emotions.

We are using a simple convolution neural network to classify the images.

Then, we are using opencv to extract 48 x 48 dimension images using the webcam and classifying them using our model.

# Requirements

The code is written in python 2.7.9 </br>
Keras (1.2.1) </br>
numpy </br>
pandas </br>
theano (0.8.2) </br>
cv2 (1.0) </br>

# Data
The data consists of scaled images from the kaggle competion kept in the data folder and also the labels in .npy format.

# Training and running the tool

To train using the data, use the script trainCNN.py </br>
Firstly, create the image data scaled, run the script </br>
python ./genScaledDat.py </br>
check if the file Scaled.bin.npy is generated in the data folder. </br>
Then run the following: </br>
python ./trainCNN.py

to run the detection using the training models run:</br>
python ./DetectEmotion.py

# Sample
Here's a snapshot from the application:</br>
</br>
![](https://github.com/DeepInEvil/webCamEmocognizer/blob/master/gif/optimised.gif)
# Todos
play around with the hyperparameters to get the best model.

# N.B.
This project was part of my Master's thesis, the thesis is also added in the documents folder.

# References:
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. <br />
Shima Alizadeh, Azar Fazel. Convolutional Neural Networks for Facial Expression Recognition. <br />
https://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/

