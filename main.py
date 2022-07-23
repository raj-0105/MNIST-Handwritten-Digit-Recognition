import tensorflow as tf
tf.random.set_seed(42)

#loading dataset
(xtrain,ytrain),(xtest,ytest)= tf.keras.datasets.mnist.load_data()

#checking the shape of dataset
xtest.shape

#checking the shape of dataset98
xtrain.shape

#checking the shape of dataset
ytest.shape

#changing label data from 1 class to 10 classes bcoz output will be on the basis of 10 classes,i.e,0-9
ytrain=tf.keras.utils.to_categorical(ytrain,num_classes=10)
ytest=tf.keras.utils.to_categorical(ytest,num_classes=10)

ytest.shape

ytrain.shape

#building the graph/model
model=tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((784,) , input_shape=(28,28,)))
model.add(tf.keras.layers.BatchNormalization())

#adding hidden layers
model.add(tf.keras.layers.Dense(200,activation='sigmoid'))
model.add(tf.keras.layers.Dense(100,activation='sigmoid'))
model.add(tf.keras.layers.Dense(60,activation='sigmoid'))
model.add(tf.keras.layers.Dense(30,activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#compiling the model
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#training the model
model.fit(xtrain,ytrain, validation_data=(xtest,ytest),epochs=30,batch_size=30)

##changing dimensions according to the models requirement using numpy
import numpy as np
np.expand_dims(xtest[0],axis=0).shape

#predicting value
prediction=model.predict(xtest[0:9])

import matplotlib.pyplot as plt
plt.imshow(xtest[4],cmap='gray')

#predicted number
predicted_num=np.argmax(prediction[4])

print(predicted_num)