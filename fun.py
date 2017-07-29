# numPos = 0
# numNeg = 0
#
# for _ in posFile:
#     numPos += 1
#
# for _ in negFile:
#     numNeg += 1
#
# images=[]
# integral=[]
# posFile.seek(0)
# negFile.seek(0)
# sum=0
# for line in posFile:
#     print line
#     line=line.split()
#     img = cv2.imread(frontDir + "croppedFrontFace/" + line[0], 0)
#     for i in range(24):
#         for j in range(24):
#             sum+=img[i][j]
#     print img
#     curr=imageDB(img, 1 , 24, 24, 1.0/(2*numPos))
#     curr.integralImage()
#     integral=curr
#     images.append(curr)
#     break
# print "################################################"
# print integral.img
# print "################################################"
# print sum







# F=0
# D=0
# for img in images:
#     result=0
#     alphaSum=0
#     for cascadeLevel in finalCascade:
#         for feature in cascadeLevel:
#             result+=feature[3]*img.testFeature(feature[0],feature[1],feature[2])
#             alphaSum += feature[3]
#     # False positive
#     if result >= 0.5*alphaSum and img.numPositive==0:
#         F+=1
#     # Real Positive
#     if result >= 0.5 * alphaSum and img.numPositive > 0:
#         D+=1
# F/=(numPos+numNeg)
# D/=(numPos+numNeg)
# while D<d*DPrev:

# fi=[]
# di=[]
# FBackup=F
# DBackup=D
# while D < d * DPrev:
#     F=FBackup
#     D=DBackup
#     for feature in finalCascade[-1]:
#         fi.append(0)
#         di.append(0)
#         for img in images:
#             if img.testFeature(feature[0], feature[1], feature[2])==1 and img.numPositive ==0:
#                 fi[-1]+=1
#             if img.testFeature(feature[0], feature[1], feature[2])==1 and img.numPositive ==1:
#                 di[-1]+=1
#         fi[-1]/=(numNeg+numPos)
#         di[-1] /= (numNeg + numPos)
#     for falsePositiveRate in fi:
#         F*=falsePositiveRate
#     for realPositiveRate in di:
#         D *= realPositiveRate
#     thresh-=threshChange