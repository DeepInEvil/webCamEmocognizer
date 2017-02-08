#script for preparing datasets, loading fer data and generating scaled images 
import pandas as pd
import numpy as np

#make a directory, keep the fer2013.csv file in the directory and also the git in the same. Otherwise, change the directory in the scripts
#load fer2013.csv

data = pd.read_csv('fer2013.csv')
data = data['pixels']
data = [ dat.split() for dat in data]
data = np.array(data)
data = data.astype('float64')
data = [[np.divide(d,255.0) for d in dat] for dat in data]

np.save('WebcamFacialExpressionRecognizer/data/Scaled.bin.npy',data)
