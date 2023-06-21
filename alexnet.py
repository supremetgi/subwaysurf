import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def alexnet(width,height):
  model = Sequential()
  model.add(Conv2D(96,(11,11),input_shape=[width,height,1],activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=2))
  model.add(Conv2D(256,(5,5),activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=2))
  model.add(Conv2D(384,(3,3),activation='relu'))
  model.add(Conv2D(384,(3,3),activation='relu'))
  model.add(Conv2D(256,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=2))
  model.add(Flatten())
  model.add(Dense(64,activation='tanh'))
  model.add(Dense(64,activation='tanh'))
  model.add(Dense(4,activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer='adam')
               

  return model




print('done')