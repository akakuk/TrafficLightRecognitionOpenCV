import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalizeImage(img):
    height, width, _ = img.shape
    print(width)
    for x in range(0, width - 1):
        for y in range(0, height - 1):
            s = img[x,y,0] + img[x,y,1] + img[x,y,2]
            if(s > 0):
                img[x,y,2] = img[x,y,2] / s
                img[x,y,1] = img[x,y,1] / s
                img[x,y,0] = img[x,y,0] / s
            else:
                img[x,y,0] = 0
                img[x,y,1] = 0
                img[x,y,2] = 0
                
    return img

img = cv2.imread("37596.png")
cv2.imshow("Slika", img);
cv2.waitKey()
cv2.destroyWindow("Slika")

img_norm = normalizeImage(img)
cv2.imshow("Normalized", img_norm);
cv2.waitKey()
cv2.destroyWindow("Normalized")



img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Hsv", img_hsv);
cv2.waitKey()
cv2.destroyWindow("Hsv")

img_blur = cv2.blur(img, (5, 5))
cv2.imshow("Blur", img_blur);
cv2.waitKey()
cv2.destroyWindow("Blur")

img_grayscale = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_grayscale);
cv2.waitKey()
cv2.destroyWindow("Grayscale")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
img_tophat = cv2.morphologyEx(img_grayscale, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Tophat", img_tophat)
cv2.waitKey()
cv2.destroyWindow("Tophat")

ret, img_globalThreshold = cv2.threshold(img_tophat, 100, 255, cv2.THRESH_BINARY)
img_adaptiveThresholdMean = cv2.adaptiveThreshold(img_tophat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
img_adaptiveThresholdGauss = cv2.adaptiveThreshold(img_tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
cv2.imshow("GlobalThreshold", img_globalThreshold)
cv2.imshow("MeanThreshold", img_adaptiveThresholdMean)
cv2.imshow("GaussThreshold", img_adaptiveThresholdGauss)
cv2.imshow("Slika", img);

cv2.waitKey()

cv2.destroyAllWindows()
cv2.waitKey()