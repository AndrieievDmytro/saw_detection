import cv2 as cv
import numpy as np
img = cv.imread("saw4.jpg", cv.IMREAD_GRAYSCALE)


scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized =  cv.resize(img, dim, interpolation = cv.INTER_AREA)

orb = cv.ORB_create(nfeatures=1500)
keypoints_orb, descriptors = orb.detectAndCompute(resized, None)
img = cv.drawKeypoints(resized, keypoints_orb, None)
cv.imshow("Image", resized)
cv.waitKey(0)
cv.destroyAllWindows()
