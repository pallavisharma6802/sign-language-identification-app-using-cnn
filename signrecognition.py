!pip install opencv-python-headless
# also contrib, if needed
!pip install opencv-contrib-python-headless
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

path="/Users/pallavisharma/Downloads/ImagePro"
files_=os.listdir(path)
files_.sort()

img_arr=[]
label_arr=[]

for i in range(len(files_)):
    subfile=os.listdir(path+"/"+files_[i])
    for j in range(len(subfile)):
        filepath=path+"/"+files_[i]+"/"+subfile[j]
        image=cv2.imread(filepath)
        image=cv2.resize(image,(100,100))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_arr.append(image)
        label_arr.append(i)

img_arr=np.array(img_arr)
label_arr=np.array(label_arr,dtype="float")

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(img_arr,label_arr,test_size=0.15)

from keras import layers,callbacks,utils,applications,optimizers
from keras.model import Sequential, Model, load_model

model=Sequential()
pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(100,100,3),include_top=False)
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.build(input_shape=(None,100,100,3))
model.summary()
