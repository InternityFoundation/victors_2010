import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import keras_preprocessing
import keras_applications
from keras_preprocessing.image import ImageDataGenerator , array_to_img, img_to_array, load_img


#DATADIR = "C:/aadhar"
#gen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True , brightness_range = (0.5, 1.5))


#path = os.path.join(DATADIR)
#for img in os.listdir(path):
 #    img1 = cv2.imread(os.path.join(path,img))
  #   new_image = cv2.resize(img1 , (150,150))
   #  img_2 = np.expand_dims(new_image, axis=0)
    # i = 0
     #for batch in gen.flow(img_2, batch_size=1,
      #                         save_to_dir='C:/new_aadhar', save_prefix='aadhar', save_format='jpg'):
       #   i += 1
        #  if i > 20:
         #      break


import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img


def detect(img, eyeCascade , faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords, img = draw_boundary(img, eyeCascade, 1.1, 10, color['blue'], "Face")
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['red'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    return img




eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, eyeCascade, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
