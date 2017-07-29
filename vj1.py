import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('/home/o/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/data/cascade.xml')

frontDir = "/home/o/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/"

posFile = open(frontDir+"frontFaceRoom/myPics.txt","r")
negFile = open(frontDir+"bg.txt","r")
n=0

for line in posFile:
    line=line.strip()
    details=line.split()
    print details
    img=cv2.imread(frontDir+details[0],0)
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    cv2.imshow("asd",img)
    for (x,y,w,h) in faces:
        n = n + 1
        cropped=img[y:(y+h) , x:(x+w)]
    #    cv2.imshow('asd', cropped)
        cropped=cv2.resize(cropped, (24,24))
    #    print img
    #    cropped=img[int(details[3]):(int(details[3])+int(details[5])) , int(details[2]):(int(details[2])+int(details[4]))]

    #    cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(frontDir + 'croppedFrontFaceRoom/img'+str(n)+'.jpeg', cropped)
# ewv9d
