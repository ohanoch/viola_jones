# import math
#import numpy as np
from viola_jones.vj2_2 import *

posFile2 = open(frontDir + "croppedFrontFaceRoom/allMyPicsRoom.info", "r")
negFile2 = open(frontDir + "croppedNegative/allBg.txt", "r")

numPos = 0
numNeg = 0

for _ in posFile2:
    numPos += 1

for _ in negFile2:
    numNeg += 1
posFile2.seek(0)
negFile2.seek(0)
allImages = []
regImages=[]
for line in posFile2:
    line = line.split()
    img = cv2.imread(frontDir + "croppedFrontFaceRoom/" + line[0], 0)
    curr = imageDB(img, 1, 24, 24, 1.0 / (2 * numPos))
    regImages.append(img)
    curr.img=curr.integralImage()
    allImages.append(curr)
posFile2.seek(0)
for line in negFile2:
    line = line.strip()
    img = cv2.imread(frontDir + "croppedNegative/" + line, 0)
    curr = imageDB(img, 0, 24, 24, 1.0 / (2 * numNeg))
    curr.img=curr.integralImage()
    allImages.append(curr)
negFile2.seek(0)

# frontDir = '/home/o/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/'
cascadeFile = open(frontDir + "finalCascadeRoom.txt", "r")
cascade=[]
for line in cascadeFile:
    line=line.strip()
    lineData=line.split(" ",2)
    cascade.append(weakClassifier([],float(lineData[0]),float(lineData[1])))
    lineFeatures=lineData[2].split(" & ")
    for stringFeature in lineFeatures:
        stringFeature=[float(n) for n in stringFeature.split()]
        for i,num in enumerate(stringFeature[0:-1]):
            stringFeature[i]=int(num)
        cascade[-1].features.append(haarFeature(stringFeature[0],stringFeature[1],stringFeature[2],stringFeature[3],stringFeature[4],stringFeature[5]))


def isFace(img):
    result = 0
    totalAlphaSum = 0
    for classifier in cascade:
        totalAlphaSum += (classifier.thresh * classifier.alphaSum)
        for feature in classifier.features:
            print str(feature.fType)+"   "+str(feature.y)+"   "+str(feature.x)+"   "+str(feature.h)+"   "+str(feature.w)
            result+=feature.alpha*img.testFeature(feature.fType, feature.y, feature.x, feature.h, feature.w)
        print str(result)+"   "+str(classifier.thresh*classifier.alphaSum)+"   "+str(totalAlphaSum)
        if not(result >= totalAlphaSum):
            print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            return False
    return True

truePos=0
trueNeg=0
for img in allImages:
    if isFace(img) and img.numPositives>0:
        truePos+=1
    if (not isFace(img)) and img.numPositives==0:
        trueNeg+=1
frame=regImages[11]
frame1=frame
frame1=cv2.resize(frame1,(640,480))
cv2.imshow("frame1", frame1)
if isFace(allImages[11]) and allImages[11].numPositives>0:
    a=0
    b=0
    for classifier in cascade:
        a+=1
        for feature in classifier.features:
            b+=1
            cv2.rectangle(frame, (feature.x,feature.y), (feature.x+feature.w,feature.y+feature.h), (a*50, 0, b*50), 1)
frame=cv2.resize(frame,(640,480))
cv2.imshow("frame", frame)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
print str(truePos)+"   "+str(trueNeg)+"   "+ str(numPos)+"   "+str(numNeg)+"   "+str(float(truePos)/numPos)+"   "+str(float(trueNeg)/numNeg)+"   "+str(
float(truePos+trueNeg)/float(numPos+numNeg))