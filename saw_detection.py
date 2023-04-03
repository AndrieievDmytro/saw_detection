import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt

# from PIL import Image

sift = cv.SIFT_create()

# im = Image.open("saw1.jpg")
# im.save("saw1_600.jpg", dpi=(600,600))

img = cv.imread('saw1.jpg', cv.IMREAD_GRAYSCALE)

scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized =  cv.resize(img, dim, interpolation = cv.INTER_AREA)

# img_orb = cv.ORB_create(nfeatures=1500)



kp, ds = sift.detectAndCompute(resized, None)

kps_img = cv.drawKeypoints(
    image     = resized, 
    keypoints = kp,
    outImage  = resized 
)

cv.imshow("Keypoints preview ", kps_img)
cv.waitKey(0)
cv.destroyAllWindows()