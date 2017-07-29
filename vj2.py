import cv2
import math
#import numpy as np

frontDir = '/home/o/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/'
posFile1 = open(frontDir + "croppedFrontFace/myPics.info", "r")
negFile1 = open(frontDir + "croppedNegative/bg.txt", "r")

kRows = 24
kCols = 24
haarValue=150

class imageDB:
    def __init__(self, img, numPositives, rows = 24, cols = 24, weight = 0.0):
        self.img = img
        self.numPositives = numPositives
        self.rows = rows
        self.cols = cols
        self.weight = weight

    def integralImage(self):
        intImage = []
        for i in range(self.rows):
            if i != 0:
                intImage.append([int(self.img[i][0]) + int(intImage[i - 1][0])])
            else:
                intImage.append([int(self.img[0][0])])
            for j in range(1,self.cols):
                if i != 0:
                    intImage[i].append(int(self.img[i][j]) + int(intImage[i][j - 1])+int(intImage[i - 1][j]) - int(intImage[i - 1][j - 1]))
                else:
                    intImage[i].append(int(self.img[i][j]) + int(intImage[i][j - 1]))
        self.img=intImage

    def calcRect(self,y,x,h,w):
        # -1 from everything because if i am on 23 and my width is 1 than 23+1=24 and I am out of bounds. essentially width of one pixel is no width at all
        bottomRight = self.img[y + h-1][x + w-1]
        if y >= 1:
            topRight = self.img[y - 1 -1][x + w-1]
            if x >= 1:
                topLeft = self.img[y - 1 -1][x - 1 -1]
                bottomLeft = self.img[y + h -1][x - 1 -1]
            else:
                topLeft = 0
                bottomLeft = 0
        else:
            topRight = 0
            topLeft = 0
            if x >= 1:
                bottomLeft = self.img[y + h -1][x - 1 -1]
            else:
                bottomLeft = 0

        return bottomRight - bottomLeft - topRight + topLeft


    def testFeature(self,featureType,h,w):
        if featureType == 1:
            #White-Black horizontal
            for y in range(kRows - h):
                for x in range(kCols - w * 2):
                    if abs(self.calcRect(y,x + w,h,w) - self.calcRect(y,x,h,w)) <= haarValue:
                        return 1
        if featureType == 2:
            #Black-White vertical
                for y in range(kRows - h * 2):
                    for x in range(kCols-w):
                        if abs(self.calcRect(y, x,h, w) - self.calcRect(y +h, x, h, w)) <= haarValue:
                            return 1
        if featureType == 3:
            #White-Black-White horizontal
            for y in range(kRows - h):
                for x in range(kCols - w * 3):
                    if abs((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) - 1 * self.calcRect(y,x + w * 2,h,w)) <= haarValue:
                        return 1
        if featureType == 4:
            #White-Black-White-Black diagonal
            for y in range(kRows - h * 2):
                for x in range(kCols - w * 2):
                    if abs((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) + self.calcRect(y + h,x,h,w) - self.calcRect(y + h,x + w,h,w)) <= haarValue:
                        return 1
        if featureType == 5:
            #White-Black-White vertica
            for y in range(kRows - h*3):
                for x in range(kCols - w):
                    if abs((- 1 )* self.calcRect(y,x,h,w) + self.calcRect(y+h,x,h,w) - 1 * self.calcRect(y+h*2,x,h,w)) <= haarValue:
                        return 1
        return 0

#This function accepts the amount of features to make a strong classifier from (T) and returns the classifier
def adaboost(T,images, currentCascade):

    for t in range(T):
        n=0
        print "adaboosting feature: " + str (t)
        #Normalizing the weights (weighing up for new feature stage)
        wSum = 0.0
        for img in images:
            wSum += img.weight
        for img in images:
            img.weight = img.weight / wSum

        minError=2
        minFeature=[-1,-1,-1]
        # Features test
        for h in range(1, kRows):
            for w in range(1, kCols):
                f1 = True
                f2 = True
                f3 = True
                f4 = True
                f5 = True
                for cascadeLevel in currentCascade:
                    for feature in cascadeLevel:
                        if feature[1]==h and feature[2]==w:
                            if feature[0]==1:
                                f1=False
                            elif feature[0]==2:
                                f2=False
                            elif feature[0]==3:
                                f3=False
                            elif feature[0]==4:
                                f4=False
                            elif feature[0]==5:
                                f5=False
                if f1 and w<kCols/2:
                #Feature1 (White-Black horizontal)
                    n+=1
                    fError = 0.0
                    for img in images:
                        yi = 0
                        if img.numPositives > 0:
                            yi = 1
                        fError += img.weight * abs(img.testFeature(1, h, w) - yi)
                    if fError < minError:
                        minError = fError
                        minFeature = [1, h, w, math.log((1-minError)/minError)]
                if f2 and h < kRows/2:
                # Feature2 (Black-White vertical)
                    n += 1
                    fError = 0.0
                    for img in images:
                        yi = 0
                        if img.numPositives > 0:
                            yi = 1
                        fError += img.weight * abs(img.testFeature(2,h, w) - yi)
                    if fError < minError:
                        minError = fError
                        minFeature = [2, h, w,math.log((1-minError)/minError)]
                if f3 and w<kCols/3:
                # Feature3 (White-Black-White)
                    n += 1
                    fError=0.0
                    for img in images:
                        yi=0
                        if img.numPositives>0:
                            yi=1
                        fError += img.weight * abs(img.testFeature(3,h,w) - yi)
                    if fError<minError:
                        minError=fError
                        minFeature=[3,h,w,math.log((1-minError)/minError)]
                if f4 and h < kRows / 2 and w < kCols/2:
                # Feature4 (White-Black-White-Black diagonal)
                    n += 1
                    fError = 0.0
                    for img in images:
                        yi = 0
                        if img.numPositives > 0:
                            yi = 1
                        fError += img.weight * abs(img.testFeature(4, h, w) - yi)
                    if fError < minError:
                        minError = fError
                        minFeature = [4, h, w,math.log((1-minError)/minError)]
                if f5 and h < kRows / 3:
                # Feature5 (White-Black-White verticle)
                    n += 1
                    fError = 0.0
                    for img in images:
                        yi = 0
                        if img.numPositives > 0:
                            yi = 1
                        fError += img.weight * abs(img.testFeature(5, h, w) - yi)
                    if fError < minError:
                        minError = fError
                        minFeature = [5, h, w,math.log((1-minError)/minError)]
        print n
        # if T==1:
        #     currentCascade.append([minFeature])
        # else:
        #     currentCascade[-1].append(minFeature)
        currentCascade[-1].append(minFeature)


        #re-weigh
        for img in images:
            img.weight *= ((minError/(1-minError))**(1-img.testFeature(minFeature[0],minFeature[1],minFeature[2])))

    return currentCascade

def createCascade(posFile,negFile,f,d,Ftarget,threshChange):
    numPos = 0
    numNeg = 0

    for _ in posFile:
        numPos += 1

    for _ in negFile:
        numNeg += 1
    posFile.seek(0)
    negFile.seek(0)
    images = []
    for line in posFile:
        line = line.split()
        img = cv2.imread(frontDir + "croppedFrontFace/" + line[0], 0)
        curr = imageDB(img, 1, 24, 24, 1.0 / (2 * numPos))
        curr.integralImage()
        images.append(curr)
    posFile.seek(0)
    for line in negFile:
        line=line.strip()
        img = cv2.imread(frontDir + "croppedNegative/" + line, 0)
        curr = imageDB(img, 0, 24, 24, 1.0 / (2 * numNeg))
        curr.integralImage()
        images.append(curr)
    negFile.seek(0)

    finalCascade=[]
    F=1.0
    D=1.0
    n1=0
    n2=0
    thresh=0.5
    while F>Ftarget:
        if len(finalCascade)>0:
            for feature in finalCascade[-1]:
                feature.append(thresh)
        thresh = 0.5
        n1+=1
        finalCascade.append([])
        FPrev = F
        while F>f*FPrev:
            DPrev = D
            print "sending to adaboost" + str(finalCascade)
            finalCascade=adaboost(1,images, finalCascade)
            n2+=1
            FPrev2=F
            while True:
                F=FPrev2
                D=DPrev
                fi=0.0
                di=0.0
                for img in images:
                    result = 0
                    alphaSum = 0
                    for feature in finalCascade[-1]:
                        result += feature[3] * img.testFeature(feature[0], feature[1], feature[2])
                        alphaSum += feature[3]
                    # print str(result) + "     " + str(alphaSum * thresh) + "    " + str(thresh) + "    " + str(feature[3]) + "    " + str(alphaSum) + "    " + str(D) + "    " + str(F) + "    " + str(di) + "    " + str(fi)
                    # print finalCascade[-1]
                    # False positive
                    if result >= thresh * alphaSum and img.numPositives == 0:
                        fi += 1
                        print "fi: " + str(fi)
                    # Real Positive
                    if result >= thresh * alphaSum and img.numPositives > 0:
                        di += 1
                        print "di: " +str(di)
                fi /=numNeg
                di /= numPos
                F*=fi
                D*=di
                print "F:" + str(F) + "    D: " + str(D) +"    "+str(thresh)+"    "+str(d*DPrev)+ "    "+ str(f*FPrev)+"   "+str(FPrev)+"    "+str(n1)+"    "+str(n2)
                if D > d * DPrev:
                    print "break"
                    break
                thresh*=threshChange
        N=[]
        if F>Ftarget:
            for img in images:
                if img.numPositives==0:
                    result=0
                    alphaSum=0
                    add=True
                    for cascadeLevel in finalCascade:
                        for feature in cascadeLevel:
                            result+=feature[3]*img.testFeature(feature[0],feature[1],feature[2])
                            alphaSum += feature[3]
                        # False positive
                        if not(result >= thresh*alphaSum):
                            add=False
                            break
                    if add:
                        N.append(img)

    # finalCascade.pop(0)
    return finalCascade

falsePositivRatePerLevel=0.3
truePPositiveRatePerLevel=0.98
falsePositiveTarget=0.005
threshChangePerLevel=0.5
with open(frontDir + "finalCascade.txt", "w") as cascadeFile:
    cascadeFile.write(str(createCascade(posFile1,negFile1,falsePositivRatePerLevel,truePPositiveRatePerLevel,falsePositiveTarget,threshChangePerLevel)))
posFile1.close()
negFile1.close()