from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from getkeys import key_check   
import cv2 as cv
import time
from alexnet import alexnet
import tensorflow as tf
import numpy as np
from PIL import ImageGrab

width,height = 80,60


model_name = 'subwaymodel.model'
#model = alexnet(width,height)
#model.load(model_name)
model = tf.keras.models.load_model(model_name)
print("you won")

path = "C:\Program Files (x86)\chromedriver"


'''
def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=(153,96,971,547)))
    cv.imshow('window',cv.cvtColor(screen,cv.COLOR_BGR2RGB))
    if cv.waitKey(25) == ord('q'):
        cv.destroyAllWindows()
    return screen
'''
        


driver = webdriver.Chrome(path)

driver.get("https://poki.com/en/g/subway-surfers")
time.sleep(4)

for i in list(range(40))[::-1]:
    time.sleep(1)
    print(i)

search = driver.find_element_by_id("game-element")



def jump():
    search.send_keys(Keys.ARROW_UP)
def right():
    search.send_keys(Keys.ARROW_RIGHT)
def left():
    search.send_keys(Keys.ARROW_LEFT)
def down():
    search.send_keys(Keys.ARROW_DOWN)




while 1:
        #search.send_keys(Keys.UP)
        #driver.execute_script("arguments[0].keydown(38);",search)
        #search.send_keys(Keys.ARROW_UP)
        screen = np.array(ImageGrab.grab(bbox=(153,96,971,547)))
        #screen = cv.cvtColor(screen,cv.COLOR_BGR2GRAY)
        screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
        screen = cv.resize(screen, (80,60))
        screen = screen/255.0
        prediction = model.predict([screen.reshape(-1,width,height,1)])[0]
        time.sleep(0.5) 
        moves = list(np.around(prediction))
        
        print(prediction,moves)
        if moves == [1,0,0,0]:
            jump()
        elif moves == [0,1,0,0]:
            left()
        elif moves == [0,0,1,0]:
            down()
        elif moves == [0,0,0,1]:
            right()
     
        

   


#TqC_a
driver.quit()