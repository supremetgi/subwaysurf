import numpy as np
#import pandas as pd
from collections import Counter
from random import shuffle
import cv2 as cv
import pandas as pd



train_data = np.load('training_data.npy',allow_pickle=True)
last = list(np.load('train_data_v2.npy',allow_pickle=True))

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

up = []
left = []
down = []
right = []


print(len(train_data))

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0,0]:
        up.append([img,choice])
    elif choice == [0,1,0,0]:
        left.append([img,choice])
    elif choice == [0,0,1,0]:
        down.append([img,choice])
    elif choice == [0,0,0,1]:
        right.append([img,choice])
  


#up = up[:len(left)][:len(right)][:len(down)]
#nil = nil[:len(up)][:len(left)][:len(right)][:len(down)]
up = up[:len(right)][:len(left)][:len(down)]
left = left[:len(up)]
right = right[:len(left)]
down = down[:len(right)]

final_data = up + left + right + down


shuffle(final_data)
print(len(final_data))

#np.save('train_data_v2.npy',final_data)
#last.append(final_data)
last = last + final_data
print(len(final_data))
print(len(last))
np.save('train_data_v2.npy',last)







