import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



def net(width,height):
  model = Sequential()
  model.add(Conv2D(64,(4,4),input_shape=[width,height,1],activation='relu'))
  #model.add(Conv2D(64, (3,3), input_shape = [width,height,1],activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))


  #model.add(Conv2D(32,(2,2)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2,2)))

  #model.add(Conv2D(32,(4,4)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2,2)))


  model.add(Flatten())
  model.add(Dense(512,activation='relu'))
  model.add(Dense(512,activation='relu'))
  model.add(Dense(512,activation='relu'))
  #model.add(Dense(64))
  #model.add(Dense(256))
  #model.add(Dense(16))
  model.add(Dense(4,activation='sigmoid'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics = ['accuracy']
                )
  #model.fit(x,y,batch_size=32,epochs=3,validation_split=0.3)
  return model




def net2(width,height):
  model  = Sequential()
  model.add(Conv2D(512,(3,3),input_shape=[width,height,1],activation='relu'))
  model.add(Activation('relu'))
  #model.add(Conv2D(64,(4,4),strides=(2,2)))
  #model.add(Activation('relu'))
  model.add(Flatten())
  #model.add(Dense(128))
  #model.add(Activation('tanh'))
  #model.add(Dense(512))
  #model.add(Activation('tanh'))
  model.add(Dense(4))
  model.add(Activation('softmax'))
  

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics = ['accuracy']
                )

  return model 



