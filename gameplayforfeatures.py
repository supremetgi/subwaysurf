from getkeys import key_check
import cv2 as cv
from PIL import ImageGrab
import os
import time
import numpy as np
"""




"""



def keys_to_output(keys):
    output = [0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    if 'A' in keys:
        output[1] = 1
    if 'S' in keys:
        output[2] = 1
    if 'D' in keys:
        output[3] = 1
   
    return output






file_name = 'training_data.npy'


if os.path.isfile(file_name):
    print("file exists print previous data")
    training_data = list(np.load(file_name,allow_pickle = True))
else:
    print("file does not exist,starting fresh")
    training_data = []


def main():
    for i in list(range(15))[::-1]:
        print(i+1)
        time.sleep(1)

    while True:
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        #cv.imshow('window',cv.cvtColor(screen,cv.COLOR_BGR2RGB))
        if cv.waitKey(25) == ord('q'):
            cv.destroyAllWindows()
            break
        screen = cv.cvtColor(screen,cv.COLOR_BGR2GRAY)
        
        screen = cv.resize(screen,(80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        if len(training_data) % 50 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
        



main()

