import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]
dataset_path='./data/'
file_name=input("Enter the name of person : ")
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    
    skip=0
    
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        
        
        if skip%10==0:
            face_data.append(face_section)
            
            
        skip=skip+1
        
    cv2.imshow("frame",frame)
    cv2.imshow("face_section",face_section)
    
    pressed_key=cv2.waitKey(1) & 0xFF
    if pressed_key==ord('q'):
        break
 #reshaping   
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
#print(face_data.shape)
#saving
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved in " +dataset_path+file_name)

cap.release()
cv2.destroyAllWindows()
 