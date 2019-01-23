# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:54:38 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# Generate train targets using previous time steps
def create_fulldata(dataset, delay):
    data, target = [], []
    for i in range(len(dataset)-delay-1):
        X = dataset[i:(i+delay),0]
        data.append(X)
        target.append(dataset[i+delay,0])
    return np.array(data), np.array(target)

# fix random seed for reproducibility
np.random.seed(7)
dataframe = pandas.read_csv('amzn_stock_small.csv', usecols=[1], engine='python')
dataframe.fillna(dataframe.mean(), inplace=True)
dataset = dataframe.values
dataset = dataset.astype('float32')
#dataframe.fillna(dataframe.mean(), inplace=True)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.84)
test_size = int((len(dataset) - train_size)) 
#valid_idx = train_size+test_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#valid = dataset[train_size:valid_idx,:]
print(len(train), len(test))

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_fulldata(train, look_back)
testX, testY = create_fulldata(test, look_back)
#valX, valY = create_fulldata(valid, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
start = time.time()
# create and fit LSTM network
batch = 32
model = Sequential()
model.add(LSTM(40, input_shape=(1,look_back),activation='tanh',dropout_U = 0.2, dropout_W = 0.2,return_sequences=True))
#model.add(LSTM(10, input_shape=(1,look_back) ,return_sequences=True))
model.add(LSTM(40, input_shape=(1,look_back),activation='tanh'))
#model.add(LSTM(4, batch_input_shape=(32, look_back, 1), stateful=True))
model.add(Dense(1))
opt = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(loss = 'mean_squared_error', optimizer=opt ,metrics=['mse'])
history = model.fit(trainX,trainY, epochs=100, batch_size=batch, verbose=2, validation_split=0.2)
end = time.time()
train_time = end - start
print('Total Train Time: %.2fs' % (train_time))
# summarize history for loss
plt.subplot(1,1,1)
plt.plot(history.history['loss'], '.-')
plt.plot(history.history['val_loss'], '.-')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.ylabel('Stock Closing Price')
plt.xlabel('Minutes Elapsed July-December 2016')
plt.legend(['real','predict-train','predict-test'], loc='upper left')
plt.show()












