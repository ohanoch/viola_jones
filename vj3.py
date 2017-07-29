from viola_jones.vj2_2 import *

frontDir = '/home/o/Desktop/Programming_Projects/Project_Onion/Onion_Excercises/Or_excercises/dataset_meir/'
cascadeFile = open(frontDir + "finalCascade.txt", "r")

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


def isFace(img,imgY,imgX,imgScaleFactor=1):
    # print "qqqqqqqqqqqqqq"
    result = 0
    totalAlphaSum = 0
    for classifier in cascade:
        totalAlphaSum += (classifier.thresh * classifier.alphaSum)
        for feature in classifier.features:
            # print str(feature.fType)+"   "+str(feature.y)+"   "+str(feature.x)+"   "+str(feature.h)+"   "+str(feature.w)
            result+=feature.alpha*img.testFeature(feature.fType, imgY+int(feature.y*imgScaleFactor), imgX+int(feature.x*imgScaleFactor),
                                                                          int(feature.h*imgScaleFactor),int(feature.w*imgScaleFactor))
        if not(result >= totalAlphaSum):
            # print str(result)+"   "+str(totalAlphaSum)
            return False
        print "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww"
    return True

font=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
scaleFactor=1.25
delta=1
# subWindowSize=19.2 #24/1.25=19.2  because loop changes value at start
# currKRows = kRows
# currKCols = kCols
# totalScaleFactor=0.8 #1/1.25=0.8  because loop changes value at start
frameCount=0
while True:
    subWindowSize = 19.2  # 24/1.25=19.2  because loop changes value at start
    currKRows = kRows/scaleFactor
    currKCols = kCols/scaleFactor
    totalScaleFactor = 0.8  # 1/1.25=0.8  because loop changes value at start
    frameCount+=1
    print "rrrr"
    # Capture frame-by-frame
    ret, frame = cap.read()


    frameRows, frameCols,_ = frame.shape


    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img=imageDB(gray, 0, frameRows, frameCols, 1)
    img.img=img.integralImage()

    while currKRows<frameRows and currKCols<frameCols:
        subWindowSize*=scaleFactor
        currKRows*= scaleFactor
        currKCols*= scaleFactor
        totalScaleFactor*=scaleFactor
        print "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz   "+str(totalScaleFactor)+"   "+ str(currKRows)+"   "+str(currKCols)
        for subWindowY in range(0,int(frameRows-round(currKRows)),int(totalScaleFactor*delta)):
            for subWindowX in range(0,int(frameCols-round(currKCols)),int(totalScaleFactor*delta)):
                # if int(totalScaleFactor)==11:
                #     print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   " \
                #           "   "+ str(currKRows)+"   "+str(currKCols)
                #     gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #     cv2.rectangle(gray2,(subWindowX,subWindowY),(subWindowX+24*int(totalScaleFactor),subWindowY+24*int(totalScaleFactor)),(255,0,0),2)
                #     cv2.imshow("gray",gray2)
                #     if cv2.waitKey(0) & 0xFF == ord('q'):
                #         cv2.destroyAllWindows()
                # subWindow=imageDB(intImg[subWindowY:subWindowY+int(round(subWindowSize)),subWindowX:subWindowX+int(round(subWindowSize))],0,subWindowSize,
                #                   subWindowSize)
                if isFace(img,subWindowY,subWindowX,totalScaleFactor):
                    print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"+str(subWindowY)+"   "+str(subWindowX)+"   "+str(frameCount)+ "   "+ str(totalScaleFactor)
                    i=0
                    for cascadeLevel in cascade:
                        for feature in cascadeLevel.features:
                            i+=50
                            cv2.rectangle(frame, (subWindowX+int(feature.x*totalScaleFactor),subWindowY+int(feature.y*totalScaleFactor)),
                                          (subWindowX+int((feature.x+feature.w)*totalScaleFactor),subWindowY+int((feature.y+feature.h)*totalScaleFactor)),
                                          (i, 255,i), 2)
                            cv2.putText(frame, 'FACE DETECTED', (100,100), font, 1, (0, 0, 255), 3, cv2.CV_AA)

    # resizeCount=0
    # tempImg=img
    # tempIntImg=intImg
    #
    # while frameRows>kRows and frameCols>kCols:
    #     tempIntImg=np.array(tempIntImg)
    #     for y in range(frameRows-kRows):
    #         for x in range(frameCols-kCols):
    #             print str(y)+"   "+str(x)+"   "+str(frameRows)+"   "+str(frameCols)+"   "+str(intImg[0][23])
    #             subWindow=imageDB(tempIntImg[y:(y+kRows),x:(x+kCols)],0,kRows,kCols,0.0)
    #             print subWindow.img.shape
    #             if isFace(subWindow):
    #                 for cascadeLevel in cascade:
    #                     for feature in cascadeLevel.features:
    #                         cv2.rectangle(frame, (frameResizeValue*resizeCount*(x+feature.x), frameResizeValue*resizeCount*(y+feature.y)), (frameResizeValue*resizeCount*(x+feature.x + feature.w), frameResizeValue*resizeCount*(y+feature.y + feature.h)), (0, 255,0), 2)
    #                         cv2.putText(frame, 'FACE DETECTED', (100,100), font, 1, (0, 0, 255), 3, cv2.CV_AA)
    #     tempImg.img = cv2.resize(tempImg.img, (int(frameRows*frameResizeValue), int(frameCols*frameResizeValue)))
    #     tempIntImg=tempImg.integralImage()
    #     frameRows, frameCols = tempImg.img.shape
    #     resizeCount+=1

    # Display the resulting frame
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&q"
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
