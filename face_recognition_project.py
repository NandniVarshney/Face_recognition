# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:10:07 2019

@author: nandn
"""

import cv2
import numpy as np
import os

#KNN ALGORITHM
def distance(X1,X2):
    return np.sqrt(((X1-X2)**2).sum())

def KNN(X,Y,qp,k=5):
    m=X.shape[0]
    dis=[]
    for i in range(m):
        d=distance(qp,X[i])
        dis.append((d,Y[i]))
    dis=sorted(dis)
    dis=np.array(dis)
    dis=dis[:k,:]
    n_val=np.unique(dis[:,1],return_counts=True)
    indice=n_val[1].argmax()
    return (n_val[0][indice])
    

#TRAINING DATA
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]
labels=[]
names={}

dataset_path='./data/'
class_id=0

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        
        #creating mapping
        names[class_id]=fx[:-4]
        
        #features
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        
        #labels
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
        
face_dataset=np.concatenate(face_data,axis=0)
face_label=np.concatenate(labels,axis=0)

#TESTING

while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for x,y,w,h in faces:
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        
        #output of KNN
        out=KNN(face_dataset,face_label,face_section.flatten())
        
        #box and text output
        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255)
        ,2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    cv2.imshow("FRAME",frame)
    pressed_key=cv2.waitKey(1) & 0xFF
    if pressed_key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
