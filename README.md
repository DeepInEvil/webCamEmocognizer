# webCamEmocognizer
A cool emotion detector using your laptop/desktop webcam

The data for this work is taken from the kaggle competiotion: 
Challenges in Representation Learning: Facial Expression Recognition Challenge https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

The dataset provided in the competion consists of gray scale images which are 48 x 48 in dimension and the corresponding labels consisting of 7 emotions.

We are using a simple convolution neural network to classify the images

Then, we are using opencv to extract 48 x 48 dimension images using the webcam and classifying them using our model.

# Requirements

Keras (1.2.1)
numpy
pandas
theano (0.8.2)
cv2 (1.0)

# Data
The data consists of scaled images from the kaggle competion kept in the data folder and also the labels in .npy format

# Training and running the tool

to train using the data, use the file trainCNN.py
python ./trainCNN.py

to run the detection using the training models run:
python ./DetectEmotion.py

# References:
ImageNet Classification with Deep Convolutional Neural Networks <br />
Convolutional Neural Networks for Facial Expression Recognition <br />
https://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/

