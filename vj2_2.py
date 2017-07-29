import cv2
import math
#import numpy as np

frontDir = '/home/or/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/'
posFile1 = open(frontDir + "croppedFrontFaceRoom/myPics.info", "r")
negFile1 = open(frontDir + "croppedNegative/bg.txt", "r")

kRows = 24
kCols = 24
haarValueTop=800
haarValueBottom=400

class weakClassifier:
    def __init__(self, features=[], alphaSum=0, thresh=0.5):
        self.features=features
        self.alphaSum=alphaSum
        self.thresh=thresh

class haarFeature:
    def __init__(self, fType, y, x, h, w,alpha):
        self.fType=fType
        self.y=y
        self.x=x
        self.h=h
        self.w=w
        self.alpha=alpha

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
        return intImage

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


    def testFeature(self,featureType,y,x,h,w):
        if featureType == 1:
            #White-Black horizontal
            if (self.calcRect(y,x + w,h,w) - self.calcRect(y,x,h,w) < haarValueTop) and\
                    (self.calcRect(y,x + w,h,w) - self.calcRect(y,x,h,w)>haarValueBottom):
                return 1
        if featureType == 2:
            #Black-White vertical
            if (self.calcRect(y, x,h, w) - self.calcRect(y +h, x, h, w) < haarValueTop) and\
                    (self.calcRect(y, x,h, w) - self.calcRect(y +h, x, h,w)<haarValueBottom):
                return 1
        if featureType == 3:
            #White-Black-White horizontal
            if ((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) - 1 * self.calcRect(y,x + w * 2,h,w) < haarValueTop) and\
                    ((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) - 1 * self.calcRect(y,x + w * 2,h,w)<haarValueBottom):
                return 1
        if featureType == 4:
            #White-Black-White-Black diagonal
            if ((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) + self.calcRect(y + h,x,h,w) - self.calcRect(y + h,x + w,h,w) < haarValueTop) and\
                    ((- 1) * self.calcRect(y,x,h,w) + self.calcRect(y,x + w,h,w) + self.calcRect(y + h,x,h,w) - self.calcRect(y + h,x + w,h,w)>haarValueBottom):
                return 1
        if featureType == 5:
            #White-Black-White vertical
            if ((- 1 )* self.calcRect(y,x,h,w) + self.calcRect(y+h,x,h,w) - 1 * self.calcRect(y+h*2,x,h,w) < haarValueTop) and\
                    ((- 1 )* self.calcRect(y,x,h,w) +self.calcRect(y+h,x,h,w) - 1 * self.calcRect(y+h*2,x,h,w)>haarValueBottom):
                return 1
        return 0

#This function accepts the amount of features to make a strong classifier from (T) and returns the classifier
def adaboost(T,images, currentCascade):

    for t in range(T):
        print "adaboosting feature: " + str (t)
        #Normalizing the weights (weighing up for new feature stage)
        wSum = 0.0
        for img in images:
            wSum += img.weight
        for img in images:
            img.weight = img.weight / wSum

        def findFeature():
            minError=2
            minFeature=haarFeature(-1,-1,-1,-1,-1,-1)
            # print str(minFeature.fType)+"   "+str(minFeature.y)+"   "+str(minFeature.x)+"   "+str(minFeature.h)+"   "+str(minFeature.w)
            n = 0
            # flag=True
            # Features test
            for y in range(kRows):
                for x in range(kCols):
                    for h in range(1, kRows+1):
                        for w in range(1, kCols+1):
                            # if y==1 and x==0 and h==21 and w==11:
                                # print "9999999999999999999999    " +  "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                            # if minFeature.fType==1 and minFeature.y==1 and minFeature.x==0 and minFeature.h==21 and minFeature.w==11 and flag:
                                # print "00000000000000000000000    " +  "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                                # flag=False
                            f1 = True
                            f2 = True
                            f3 = True
                            f4 = True
                            f5 = True
                            for cascadeLevel in currentCascade:
                                for feature in cascadeLevel.features:
                                    if feature.x==x and feature.y==y and feature.h==h and feature.w==w:
                                        if feature.fType==1:
                                            f1=False
                                            # print "f1   "+str(f1)
                                        if feature.fType==2:
                                            f2=False
                                            # print "f2   " + str(f2)
                                        if feature.fType==3:
                                            f3=False
                                            # print "f3   " + str(f3)
                                        if feature.fType==4:
                                            f4=False
                                            # print "f4   " + str(f4)
                                        if feature.fType==5:
                                            f5=False
                                            # print "f5   " + str(f5)
                                        # print "qqqqqqqq    "+ str(feature.fType)+"   "+str(feature.y)+"   "+str(feature.x)+"   "+str(feature.h)+"   "+str(feature.w)
                                        # print "wwwwwwww    "+str(minFeature.fType) + "   " + str(minFeature.y) + "   " + str(minFeature.x) + "   " + str(minFeature.h) + "   " + str(minFeature.w)
                            if y+h<=kRows:
                                if w <= (kCols-x) / 2:
                                #Feature1 (White-Black horizontal)
                                    # print "1111111111" + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                                    if f1:
                                        # if y==1 and x==0 and h==21 and w==11:
                                            # print "ggggggggggggg    "+str(f1)+"   "+str(y)+"   "+str(x)+"   "+str(h)+"   "+str(w)
                                        n+=1
                                        fError = 0.0
                                        for img in images:
                                            yi = 0
                                            if img.numPositives > 0:
                                                yi = 1
                                            fError += img.weight * abs(img.testFeature(1, y, x, h, w) - yi)
                                        if fError < minError:
                                            minError = fError
                                            minFeature = haarFeature(1,y,x,h,w,math.log((1-minError)/minError))
                                            # if y == 1 and x == 0 and h == 21 and w == 11:
                                                # print "1111111111111111111111    " + str(f1) + "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)

                                    if f3 and w <= (kCols - x) / 3:
                                        # Feature3 (White-Black-White)
                                        # print "33333333333" + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                                        n += 1
                                        fError = 0.0
                                        for img in images:
                                            yi = 0
                                            if img.numPositives > 0:
                                                yi = 1
                                            fError += img.weight * abs(img.testFeature(3, y, x, h, w) - yi)
                                        if fError < minError:
                                            minError = fError
                                            minFeature = haarFeature(3, y, x, h, w, math.log((1 - minError) / minError))
                                            # if y == 1 and x == 0 and h == 21 and w == 11:
                                            #     print "33333333333333333333    " + str(f1) + "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)

                            if x+w<=kCols:
                                if h <= (kRows-y)/2:
                                # Feature2 (Black-White vertical)
                                #     print "2222222222"+str(y)+"   "+str(x)+"   "+str(h)+"   "+str(w)
                                    if f2:
                                        n += 1
                                        fError = 0.0
                                        for img in images:
                                            yi = 0
                                            if img.numPositives > 0:
                                                yi = 1
                                            fError += img.weight * abs(img.testFeature(2, y, x, h, w) - yi)
                                        if fError < minError:
                                            minError = fError
                                            minFeature = haarFeature(2, y, x, h, w,math.log((1-minError)/minError))
                                            # if y == 1 and x == 0 and h == 21 and w == 11:
                                            #     print "222222222222222222222    " + str(f1) + "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)


                                    if f5 and h <= (kRows - y) / 3:
                                        # Feature5 (White-Black-White verticle)
                                        # print "555555555555" + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                                        n += 1
                                        fError = 0.0
                                        for img in images:
                                            yi = 0
                                            if img.numPositives > 0:
                                                yi = 1
                                            fError += img.weight * abs(img.testFeature(5, y, x, h, w) - yi)
                                        if fError < minError:
                                            minError = fError
                                            minFeature = haarFeature(5, y, x, h, w, math.log((1 - minError) / minError))
                                            # if y == 1 and x == 0 and h == 21 and w == 11:
                                            #     print "55555555555555555    " + str(f1) + "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)


                            if f4 and h <= (kRows-y) / 2 and w <= (kCols - x) /2:
                            # Feature4 (White-Black-White-Black diagonal)
                            #     print "444444444444" + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)
                                n += 1
                                fError = 0.0
                                for img in images:
                                    yi = 0
                                    if img.numPositives > 0:
                                        yi = 1
                                    fError += img.weight * abs(img.testFeature(4, y, x, h, w) - yi)
                                if fError < minError:
                                    minError = fError
                                    minFeature = haarFeature(4, y, x, h, w,math.log((1-minError)/minError))
                                    # if y == 1 and x == 0 and h == 21 and w == 11:
                                    #     print "444444444444444444444444    " + str(f1) + "   " + str(y) + "   " + str(x) + "   " + str(h) + "   " + str(w)

            print n
            # print "yyyyyyyyyyyyyyyy"+str(minFeature.fType) + "   " + str(minFeature.y) + "   " + str(minFeature.x) + "   " + str(minFeature.h) + "   " + str(minFeature.w)
            return (minFeature,minError)
        # if T==1:
        #     currentCascade.append([minFeature])
        # else:
        #     currentCascade[-1].append(minFeature)
        nextFeature=findFeature()
        # print str(nextFeature[0].fType) + "   " + str(nextFeature[0].y) + "   " + str(nextFeature[0].x) + "   " + str(nextFeature[0].h) + "   " + str(nextFeature[0].w)
        (currentCascade[-1].features).append(nextFeature[0])


        #re-weigh
        for img in images:
            img.weight *= ((nextFeature[1]/(1-nextFeature[1]))**(1-img.testFeature(nextFeature[0].fType,nextFeature[0].y,nextFeature[0].x,nextFeature[0].h,nextFeature[0].w)))

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
    posImages = []
    for line in posFile:
        line = line.split()
        img = cv2.imread(frontDir + "croppedFrontFace/" + line[0], 0)
        curr = imageDB(img, 1, 24, 24, 1.0 / (2 * numPos))
        curr.img=curr.integralImage()
        posImages.append(curr)
    posFile.seek(0)
    negImages = []
    for line in negFile:
        line=line.strip()
        img = cv2.imread(frontDir + "croppedNegative/" + line, 0)
        curr = imageDB(img, 0, 24, 24, 1.0 / (2 * numNeg))
        curr.img=curr.integralImage()
        negImages.append(curr)
    negFile.seek(0)

    finalCascade=[]
    F=1.0
    D=1.0
    n1=0
    n2=0
    while F>Ftarget:
        images=posImages+negImages
        newWeakClassifier = weakClassifier()
        print newWeakClassifier.features
        newWeakClassifier.features=[]
        n1+=1
        print "testing"
        for cascadeLevel in finalCascade:
            for feature in cascadeLevel.features:
                print str(feature.fType) + "   " + str(feature.y) + "   " + str(feature.x) + "   " + str(feature.h) + "   " + str(feature.w)
            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        finalCascade.append(newWeakClassifier)
        FPrev = F
        while F>f*FPrev:
            DPrev = D
            print "sending to adaboost"
            for cascadeLevel in finalCascade:
                for feature in cascadeLevel.features:
                    print str(feature.fType) +"   "+str(feature.y) +"   "+str(feature.x) +"   "+str(feature.h) +"   "+str(feature.w)
                print "**********************"
            finalCascade=adaboost(1,images, finalCascade)
            print "got from adaboost"
            for cascadeLevel in finalCascade:
                for feature in cascadeLevel.features:
                    print str(feature.fType) + "   " + str(feature.y) + "   " + str(feature.x) + "   " + str(feature.h) + "   " + str(feature.w)
                print "&&&&&&&&&&&&&&&&&&&&&&&"
            n2+=1
            FPrev2=F
            while True:
                F=FPrev2
                D=DPrev
                fi=0.0
                di=0.0
                for img in images:
                    result = 0
                    finalCascade[-1].alphaSum=0
                    for feature in finalCascade[-1].features:
                        result += feature.alpha * img.testFeature(feature.fType, feature.y, feature.x, feature.h, feature.w)
                        finalCascade[-1].alphaSum += feature.alpha
                    # print str(result) + "     " + str(alphaSum * thresh) + "    " + str(thresh) + "    " + str(feature[3]) + "    " + str(alphaSum) + "    " + str(D) + "    " + str(F) + "    " + str(di) + "    " + str(fi)
                    # print finalCascade[-1].alphaSum
                    # False positive
                    if result >= finalCascade[-1].thresh * finalCascade[-1].alphaSum and img.numPositives == 0:
                        fi += 1
                    # Real Positive
                    if result >= finalCascade[-1].thresh * finalCascade[-1].alphaSum and img.numPositives > 0:
                        di += 1
                print "fi: " + str(fi)+"   di: " +str(di)
                fi /=numNeg
                di /= numPos
                F*=fi
                D*=di
                print "F:" + str(F) + "    D: " + str(D) +"    "+str(finalCascade[-1].thresh)+"    "+str(d*DPrev)+ "    "+ str(f*FPrev)+"   "+str(FPrev)+"    "+str(n1)+"    "+str(n2)
                if D > d * DPrev or len(finalCascade[-1].features)==1:
                    print "break"
                    break
                finalCascade[-1].thresh*=threshChange
        tempNegImages=[]
        if F>Ftarget:
            for img in negImages:
                result=0
                totalAlphaSum=0
                add=True
                for cascadeLevel in finalCascade:
                    totalAlphaSum+=(cascadeLevel.thresh*cascadeLevel.alphaSum)
                    for feature in cascadeLevel.features:
                        result += feature.alpha * img.testFeature(feature.fType, feature.y, feature.x, feature.h,feature.w)
                    # False positive
                    if not(result >= totalAlphaSum):
                        add=False
                        break
                if add:
                    tempNegImages.append(img)
        negImages=tempNegImages
        print "NNNNNNNNNNNNNNNNNN   " +str(len(negImages))

    # finalCascade.pop(0)
    return finalCascade


if __name__ == "__main__":
    falsePositivRatePerLevel=0.5
    truePPositiveRatePerLevel=0.98
    falsePositiveTarget=0.001
    threshChangePerLevel=0.5
    cascade=createCascade(posFile1,negFile1,falsePositivRatePerLevel,truePPositiveRatePerLevel,falsePositiveTarget,threshChangePerLevel)
    with open(frontDir + "finalCascadeRoom.txt", "w") as cascadeFile:
        for cascadeLevel in cascade:
            cascadeFile.write(str(cascadeLevel.alphaSum)+" "+str(cascadeLevel.thresh)+" ")
            for feature in cascadeLevel.features[0:-1]:
                cascadeFile.write(str(feature.fType)+" "+str(feature.y)+" "+str(feature.x)+" "+str(feature.h)+" "+str(feature.w)+" "+str(feature.alpha)+" & ")
            cascadeFile.write(str(cascadeLevel.features[-1].fType)+" "+str(cascadeLevel.features[-1].y)+" "+str(cascadeLevel.features[-1].x)+" "+str(cascadeLevel.features[-1].h)+" "+str(cascadeLevel.features[-1].w)+" "+str(cascadeLevel.features[-1].alpha)+"\n")

    posFile1.close()
    negFile1.close()