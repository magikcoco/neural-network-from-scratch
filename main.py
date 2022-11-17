import numpy as np #for matrix math
import pandas as pd #for importing dataset
from matplotlib import pyplot as plt #for plotting data

##importing and organizing the data
raw_data = pd.read_csv("mnist-csv/train.csv") #training data
raw_data = np.array(raw_data) #want numpy array instead of pandas dataframe
m, n = raw_data.shape #dimensions of the data, m is rows and n is features+1
np.random.shuffle(raw_data) #shuffle the data randomly around
data_dev = raw_data[0:1000].T #transpose the first 1000 data
Xdev = data_dev[0]
Ydev = data_dev[1:n]
training_data = raw_data[1000:m].T #transpose the last piece of data
Xtrain = training_data[1:n]
Ytrain = training_data[0]

def initialize_paramaters():
    w1 = np.random.rand(10,784)-0.5 #random values between 0 and 1
    b1 = np.random.rand(10,1)-0.5
    w2 = np.random.rand(10, 784) - 0.5  # random values between 0 and 1
    b2 = np.random.rand(10, 1) - 0.5
    return w1,b1,w2,b2

def ReLu(z):
    return np.maximum(0,z)

def forward_propogation(w1,b1,w2,b2,x):
    z1 = w1.dot(x) + b1
    a1 = ReLu(z1)

##WIP
