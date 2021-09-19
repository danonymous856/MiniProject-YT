import cv2
import numpy as np
import face_recognition

imgX=face_recognition.load_image_file('images/jack ma.jpg')
imgX=cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)

imgTest=face_recognition.load_image_file('images/jeff_bezos.jpg')
imgTest=cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgX)[0]
encodeX=face_recognition.face_encodings(imgX)[0]
cv2.rectangle(imgX, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results=face_recognition.compare_faces([encodeX],encodeTest)
faceDis=face_recognition.face_distance([encodeX],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('justin bieber',imgX)
cv2.imshow('justin bieber',imgTest)
cv2.waitKey(0)