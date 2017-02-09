import pandas as pd
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, RMSprop

np.random.seed(2222)  # for reproducibility

#Load the scaled data, both pixels and labels
X_train = np.load('./data/Scaled.bin.npy')
Y_tr_labels = np.load('./data/labels.bin.npy')

#reshape the given pixels into 48 X 48 images
shapex , shapey = 48, 48
X_train = X_train.reshape(X_train.shape[0] ,  shapex , shapey,1)

#convert labels to one-hot-encoding
Y_tr_labels = np_utils.to_categorical(Y_tr_labels)

#define the model 32 filters in first convolution layer followed by a max pooling and dense layer with dropout (50%)
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,init='lecun_uniform'))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

#training the model with cross sgd and nesterov momentum

sgd = SGD(lr=0.055, decay=1e-6, momentum=0.9, nesterov=True)
#optm = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train,Y_tr_labels , batch_size=128 , nb_epoch=15)

#save the model weights
import h5py
json_string = model.to_json()
model.save_weights('./models/Face_model_weights.h5')
open('./models/Face_model_architecture.json', 'w').write(json_string)
model.save_weights('./models/Face_model_weights.h5')


