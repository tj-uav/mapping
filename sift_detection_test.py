import cv2
import numpy as np


img = cv2.imread('C:\\Users\jaspe\Documents\Github_Local\mapping-tjuav\ImgSampleD\\369.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints_3.jpg',img)