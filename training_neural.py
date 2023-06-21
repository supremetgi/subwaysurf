#import matplotlib.pyplot as plt
import numpy as np
from alexnet import alexnet
width,height,lr = 80,60,1e-3
path = 'train_data_v2.npy'
#/content/drive/My Drive/train_data_v2.npy
model = alexnet(width,height)
train_data = np.load(path,allow_pickle = True)
print(len(train_data))
value = int(0.95*len(train_data))
train = train_data[:value]
#test = train_data[-50:]
#print(len(train))
#plt.imshow(train_data[600][0])
print(len(train))

x = np.array([i[0]/255.0 for i in train])
x.reshape(len(train), width, height, 1)

y = [i[1] for i in train]
#print(len(x))

y = np.array(y)
#x = np.array(x)
model.fit(x,y,batch_size=32,epochs=5,validation_split = 0.2)
model.save('subwaymodel.model')
