#import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
#from alexnet import alexnet
#from net import net
#from recurrentnet import net
#from net import net
import time
import random
width = 160
height  = 120
path = '/content/drive/My Drive/train_data_v4.npy'


x = []
y = []


model = net2(width,height)
#model = net2(width,height)

train_data  = list(np.load(path,allow_pickle = True))

random.shuffle(train_data)

k = []
for i in train_data:
  if i[1] != [0,0,0,0]:
          k.append(i)

for i in k:
    
        pic = i[0].reshape(160,120,1)
        label = i[1]
        x.append(pic)
        y.append(label)
   
    
print(len(x))
print(len(y))
time.sleep(5)
x  = np.array(x)
print(x.shape)
x =  np.reshape(x,(-1,160,120,1))
print(x.shape)
y = np.array(y)
print(y.shape)


print('dfdf')
model.fit(x,y,batch_size=50,epochs=15,validation_split = 0.1)

model.save(f"/content/drive/My Drive/subwaymodel{i}.model")