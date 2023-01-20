import cv2
import numpy as np
import os
import face_recognition

imghit = face_recognition.load_image_file('img/gandhi3.jpg')
imghit = cv2.cvtColor(imghit,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('img/gandhi.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imghit)[0]
encodeHit = face_recognition.face_encodings(imghit)[0]
cv2.rectangle(imghit,(faceloc[3],faceloc[0],faceloc[1],faceloc[2]),(255,0,255),2)

face_location_gandhi = face_recognition.face_locations(imgTest)[0]
encode_gandhi = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (face_location_gandhi[3], face_location_gandhi[0], face_location_gandhi[1], face_location_gandhi[2]), (255, 0, 255), 2)


results = face_recognition.compare_faces([encodeHit], encode_gandhi)
faceDis = face_recognition.face_distance([encodeHit],encode_gandhi)
print(results)
print(faceDis)
cv2.putText(imghit,f'{results}{round(faceDis[0],2)}',(30,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)



cv2.imshow('gandhi 3', imghit)
cv2.imshow('mahatma gandhi ', imgTest)
cv2.waitKey(0)


