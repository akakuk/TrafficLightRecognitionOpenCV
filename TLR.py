import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalizeImage(img):
    width, height, _ = img.shape
    print(width)
    for x in range(0, width - 1):
        for y in range(0, height - 1):
            s = np.int16(img[x,y,0]) + np.int16(img[x,y,1]) + np.int16(img[x,y,2])
            if(s > 0):
                img[x,y,2] = np.uint8((img[x,y,2] / s) * 255)
                img[x,y,1] = np.uint8((img[x,y,1] / s) * 255)
                img[x,y,0] = np.uint8((img[x,y,0] / s) * 255)
            else:
                img[x,y,0] = np.uint8(0)
                img[x,y,1] = np.uint8(0)
                img[x,y,2] = np.uint8(0)
    return img

def filterImg(img, Ch1Min, Ch1Max, Ch2Min, Ch2Max, Ch3Min, Ch3Max):
    blueCh, greenCh, redCh  = cv2.split(img)
    _, blueMask = cv2.threshold(blueCh, Ch1Min, Ch1Max, cv2.THRESH_BINARY)
    _, greenMask = cv2.threshold(greenCh, Ch2Min, Ch2Max, cv2.THRESH_BINARY)
    _, redMask = cv2.threshold(redCh, Ch3Min, Ch3Max, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(blueMask, mask=greenMask)
    mask = cv2.bitwise_and(mask, mask=redMask)
    return mask


img = cv2.imread("37596.png")
cv2.imshow("Slika", img)
cv2.waitKey()
cv2.destroyWindow("Slika")

img_norm = normalizeImage(img)
cv2.imshow("Normalized", img_norm)
cv2.waitKey()
cv2.destroyWindow("Normalized")

mask1 = filterImg(img_norm, 0, 150, 0, 150, 150, 255)
mask2 = filterImg(img_norm, 0, 150, 150, 255, 150, 255)
mask3 = filterImg(img_norm, 0, 150, 240, 255, 220, 255)
mask = cv2.bitwise_or(mask1, maskk=mask2)
mask = cv2.bitwise_or(mask, mask=mask3)
mask_final = cv2.merge((mask, mask, mask))

img_filtered = cv2.bitwise_and(img, mask=mask_final)
cv2.imshow("Filter", mask)
cv2.waitKey()
cv2.destroyWindow("Filter")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Hsv", img_hsv)
cv2.waitKey()
cv2.destroyWindow("Hsv")

img_blur = cv2.blur(img_norm, (5, 5))
cv2.imshow("Blur", img_blur)
cv2.waitKey()
cv2.destroyWindow("Blur")

img_grayscale = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_grayscale)
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
cv2.imshow("Slika", img)

cv2.waitKey()

cv2.destroyAllWindows()
cv2.waitKey()