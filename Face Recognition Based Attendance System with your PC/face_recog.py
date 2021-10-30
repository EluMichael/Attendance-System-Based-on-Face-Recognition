import cv2
from face_detection import face
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import sys
from embedding import emb

label=None
classes = os.listdir('Images/')
names = pd.factorize(classes, sort=True)[1]
unique_classes = np.unique(pd.factorize(classes, sort=True)[0])
people={key: value for (key, value) in zip(unique_classes, names)}
e=emb()
fd=face()

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

model=load_model('face_reco1.MODEL')

print('Recognizing face ')

while cv2.waitKey(1) != 27:
    ret,frame=source.read()
    frame=cv2.flip(frame,1)
    det,coor=fd.detectFace(frame)

    if det is not None:
        for i in range(len(det)):
            detected=det[i]
            k=coor[i]
            f=detected
            detected=cv2.resize(detected,(160,160))
            detected=detected.astype('float')/255.0
            detected=np.expand_dims(detected,axis=0)
            feed=e.calculate(detected)
            feed=np.expand_dims(feed,axis=0)
            prediction=model.predict(feed)[0]
            
            print(model.predict(feed))

            result=int(np.argmax(prediction))
            if np.max(prediction)>0.95:
                for i in people:
                    if(result==i):
                        label=people[i]
            else:
                label='unknown'

            cv2.putText(frame,label,(k[0],k[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.rectangle(frame,(k[0],k[1]),(k[0]+k[2],k[1]+k[3]),(252,160,39),3)
            cv2.imshow('onlyFace',f)
    cv2.imshow('frame',frame)

source.release()
cv2.destroyAllWindows()