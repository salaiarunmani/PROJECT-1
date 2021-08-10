from keras.models import load_model #used to load the .h5 file
from time import sleep
from keras.preprocessing.image import img_to_array #converting image to array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'H:\6th sem\project\haarcascade_frontalface_default.xml') #used to detect the face
classifier =load_model(r'H:\6th sem\project\Emotion_little_vgg.h5')#calling the trained file

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0) #web camera



while True:
    # Grab a single frame of video
    ret, frame = cap.read() #reading the camera
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert colour input to gray color
    faces = face_classifier.detectMultiScale(gray,1.3,5)#scale down our i/p image (pixel down)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#2 is tickness of the rectangle BGR colour rectagle
        roi_gray = gray[y:y+h,x:x+w] #roi=reasong of interest is our face in this program
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #resize our input
    # rect,face,image = face_detector(frame)

#60% happy 40%-angry = happy 
        if np.sum([roi_gray])!=0: #numpy sum not equal to 0 do this used to through the error face is not detected
            roi = roi_gray.astype('float')/255.0 #to reduce the pixel size max 255 min 0/1
            roi = img_to_array(roi) #roi=reason of interest is face.
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class roi=reason of interest

            preds = classifier.predict(roi)[0] #random value predit roi 0-random values first index
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3) #BGR= green colour 255
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    #20,60 are position
    cv2.imshow('Emotion Detector',frame) #show the frame as Emo Det
    if cv2.waitKey(1) & 0xFF == ord('q'): #q = quit.by pressing q we quit the program/frame
        break

cap.release()
cv2.destroyAllWindows()
