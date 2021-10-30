import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
from ann import Architecture
from embedding import emb
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

classes = os.listdir('Images/')
n_classes = len(classes)
e=emb()
arc=DenseArchs(n_classes)
face_model=arc.arch()

x_data=[]
y_data=[]

learning_rate=0.01
epochs=27
batch_size=32

for label in classes:
    for i in os.listdir('Images/'+label):
        img=cv2.imread('Images'+'/'+label+'/'+i,1)
        img=cv2.resize(img,(160,160)) # resize image
        img=img.astype('float')/255.0
        img=np.expand_dims(img,axis=0)
        embs=e.calculate(img) # apply facenet embedding

        x_data.append(embs)
        print('\n')
        print(embs.shape)
        
        y_data.append(label)

x_data = np.array(x_data,dtype='float')
# Factorizing the labels
labels = pd.factorize(y_data, sort=True)
y_data = labels[0]
y_data = np.array(y_data)

y_data = y_data.reshape(len(y_data),1)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=77, stratify=y_data)
y_train = to_categorical(y_train,num_classes=n_classes)
y_test = to_categorical(y_test,num_classes=n_classes)
print(y_test)

opt = Adam(lr=learning_rate,decay=learning_rate/epochs)
face_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
face_model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle=True,validation_data=(x_test,y_test))
print(face_model.evaluate(x_test, y_test))
face_model.save('face_reco1.MODEL')

#!! Todo:
# *Use tensorflow loader 
# *Make sure classes are conrespoding to numeric label.
# *Use OpenCV tensorflow model loader for demo.