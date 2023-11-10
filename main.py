#importing libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import cv2 
from matplotlib import pyplot as plt
import os
import random
from PIL import Image

#creating dataframe for dataset

df = pd.read_csv("C:\\Users\\duttd\\Documents\\VS_code_docs\\Google_land_mark_detection\\train.csv")
base_path='./images/'
sample=20000 #working on small samples due to hardware restriction
df = df.loc[df['if'].str.startswith('00',na=False),:]
num_classes = len(df["landmark_id"].unique()) #catagorizing w.r.t landmark ID
num_data=len(df)

#creating workable dataset

data = pd.DataFrame(df["landmark_id"].value_counts())
data.reset_index(inplace=True)
data.columns=['landmark_id','count']

#visual representation
plt.hist(data['count'],100,range=(0,58),label='visual representation of no. of images')
plt.show()
data['count'].describe()

plt.hist(df['landmark_id'],bins=df['landmark_id'].unique())
plt.show()

# training model:
#1.) label encoder

from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df['landmark_id'])

def encode_label(lbl):
    return lencoder.transform(lbl)

def decode_label(lbl):
    return lencoder.inverse_transform()

def get_image_from_number(num):
    fname, label=df.iloc[num,:]
    fname = fname +'.jpg'
    f1=fname[0]
    f2=fname[1]
    f3=fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(base_path,path))
    return im,label

print("some sample images from random classes")
fig=plt.figure(figsize=(16,16))
for i in range(1,5):
    ri=random.choices(os.listdir('base_path'),k=3)
    folder=base_path + '/' + ri[0] + '/' +ri[1] + '/' +ri[2]
    random_img= random.choice(os.listdir(folder))
    img=np.array(Image.open(folder+'/'+random_img))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis('off')
plt.show()

#2.)model building

from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras import Sequential
tf.compat.v1.disable_eager_execution()

learning_rate=0.0001
decay_speed=1e-6
momentum =0.09
loss_function ="sparse_categorical_crossentropy"
source_model=VGG19(weights=None)
drop_layer=Dropout(0.5)
drop_layer2=Dropout(0.5)

model=Sequential()
for layer in source_model.layers[:-1]:
    if layer==source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes,activation="softmax"))
model.summary()

optim1=keras.optimizers.RMSprop(lr=learning_rate)
model.compile(optimizer=optim1,
              loss=loss_function,
              metrics=["accuracy"])

def image_reshape(im, target_size):
    return cv2.resize(im,target_size)

def get_batch(dataframe,start,batch_size):
    image_array=[]
    label_array=[]

    end_img=start+batch_size
    if (end_img) > len(dataframe):
        end_img=len(dataframe)
    for idx in range(start,end_img):
        n=idx
        im,label=get_image_from_number(n,dataframe)
        im=image_reshape(im,(224,224))/255.0
        image_array.append(im)
        label_array.append(label)
    label_array=encode_label(label_array)
    return np.array(image_array), np.array(label_array)

batch_size=10
epochs_shuffle=True
weight_classes=True
epochs=1

train,val=np.split(df.sample(frac-1),[int(0.8*len(df))])
print(len(train),len(val))

for e in range(epochs):
    print("epochs: " +str(e+1) + '/' +str(epochs))
    if epochs_shuffle:
        train=train.shmple(frac-1)
    for it in range(int(np.ceil(len(train)/batch_size))):
        x_train,y_train=get_batch(train,it*batch_size,batch_size)
        model.train_on_batch(x_train,y_train)
model.save("Model")

batch_size=16
errors=0
good_preds=[]
bad_preds=[]

for it in range(int(np.ceil(len(val)/batch_size))):
    x_val,y_val=get_batch(val,it*batch_size,batch_size)
    result=model.predict(x_val)
    cla = np.argmax(result,axis-1)
    for idx,res in enumerate(result):
        if cla[idx] != y_val[idx]:
            errors+=1
            bad_preds.append([batch_size*it + idx],cla[idx],res[cla[idx]])
        else:
            good_preds.append([batch_size*it + idx],cla[idx],res[cla[idx]])


for i in range(1,6):
    n=int(good_preds[i])
    img ,lbl=get_image_from_number(n,val)
    img=cv2.cvtcolor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    