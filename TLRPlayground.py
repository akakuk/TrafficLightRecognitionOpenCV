import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalizeImage(img):
    width, height, _ = img.shape
    print(width)
    for x in range(0, width - 1):
        for y in range(0, height - 1):
            s = np.int16(img[x,y,0]) + np.int16(img[x,y,1]) + np.int16(img[x,y,2])
            s=np.float64(s)
            if(s > 0):
                img[x,y,2] = (img[x,y,2] / s) * 255
                img[x,y,1] = (img[x,y,1] / s) * 255
                img[x,y,0] = (img[x,y,0] / s) * 255
            else:
                img[x,y,0] = 0
                img[x,y,1] = 0
                img[x,y,2] = 0
    return cv2.normalize( img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def filterImg(img, Ch1Min, Ch1Max, Ch2Min, Ch2Max, Ch3Min, Ch3Max):
    blueCh, greenCh, redCh  = cv2.split(img)
    blueMask = cv2.inRange(redCh, Ch1Min, Ch1Max)
    greenMask = cv2.inRange(greenCh, Ch2Min, Ch2Max)
    redMask = cv2.inRange(blueCh, Ch3Min, Ch3Max)
    cv2.imshow("blue", redMask)
    cv2.imshow("green", greenMask)
    cv2.imshow("red", blueMask)

    maskTemp = cv2.bitwise_and(blueMask, greenMask)
    cv2.imshow("temp", maskTemp)
    mask = cv2.bitwise_and(maskTemp, redMask)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return mask 

def ROIHistogram(img):
    r = cv2.selectROI(img)
    img_cropRGB = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    img_cropHSV = cv2.cvtColor(img_cropRGB, cv2.COLOR_BGR2HSV)
    cv2.imshow("Crop",img_cropRGB)
    cv2.waitKey()
    cv2.destroyAllWindows()
    b, g, r = cv2.split(img_cropRGB)
    h, s, v = cv2.split(img_cropHSV)
    plt.figure("Hist", figsize=(6.4,2*4.8))
    plt.subplot(4,1,1)
    plt.hist(np.array(b).flatten(), bins=20, color="b")
    plt.subplot(4,1,2)
    plt.hist(np.array(g).flatten(), bins=20, color="g")
    plt.subplot(4,1,3)
    plt.hist(np.array(r).flatten(), bins=20, color="r")
    plt.subplot(4,1,4)
    plt.hist(np.array(h).flatten(), bins=14, color="b")
    #plt.hist(np.array(s).flatten(), bins=10, color="g")
    #plt.hist(np.array(v).flatten(), bins=10, color="r")
    plt.show("Hist")
    
 
img = cv2.imread("37596.png")
#img = cv2.imread("15096.png")
#img = cv2.imread("22008.png")
#img = cv2.imread("14888.png")
cv2.imshow("Slika", img)
cv2.waitKey()
cv2.destroyWindow("Slika")
originalImg = img.copy()
#width, height, channel = img.shape
#print(img.shape[1])
img = img[0:int(img.shape[0]/2), 0:int(img.shape[1]), 0:3].copy()
 
#ROIHistogram(img)

#img_norm = normalizeImage(img)
#cv2.imshow("Normalized", img_norm) 
#cv2.waitKey()
#cv2.destroyWindow("Normalized")

#ROIHistogram(img_norm)

img_blur = cv2.blur(img, (5, 5))
cv2.imshow("Blur", img_blur)
cv2.waitKey()
cv2.destroyWindow("Blur")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Hsv", img_hsv)
cv2.waitKey()
cv2.destroyWindow("Hsv")

img_hsv_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
cv2.imshow("Hsv", img_hsv_blur)
cv2.waitKey()
cv2.destroyWindow("Hsv")

mask1 = filterImg(img_blur , 25, 75, 100, 255, 0, 150)
mask2 = filterImg(img_hsv_blur, 100, 255, 100, 255, 50, 80)
mask3 = filterImg(img_blur, 120, 255, 15, 180, 0, 100)
mask4 = filterImg(img_hsv_blur, 100, 255, 100, 255, 0, 25)
maskTemp1 = cv2.bitwise_or(mask1, mask2)
maskTemp2 = cv2.bitwise_or(mask3, mask4)
mask_final = cv2.bitwise_or(maskTemp1, maskTemp2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
mask_erode = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
mask_erode = cv2.morphologyEx(mask_erode, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Mask1", mask1)
cv2.imshow("Mask2", mask2)
cv2.imshow("Mask3", mask3)
cv2.imshow("Mask4", mask4)
cv2.imshow("Erode", mask_erode)

img_filtered = cv2.bitwise_and(img, img, mask=mask_erode)
cv2.imshow("Filter", img_filtered)
cv2.waitKey()
cv2.destroyWindow("Filter")

img_grayscale = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_grayscale)
cv2.waitKey()
cv2.destroyWindow("Grayscale")

ret, img_globalThreshold = cv2.threshold(img_grayscale, 15, 255, cv2.THRESH_BINARY)
cv2.imshow("GlobalThreshold", img_globalThreshold)
cv2.waitKey()
cv2.destroyAllWindows()

circles = cv2.HoughCircles(img_globalThreshold , cv2.HOUGH_GRADIENT, 1.2, 100, param1=255, param2=8, minRadius=1, maxRadius=20)
#print(circles)
#tlr_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
ROIs = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
        ROIs.append(img[i[1]-i[2]*8:i[1]+i[2]*8, i[0]-i[2]*4:i[0]+i[2]*4, 0:3])
        #cv2.rectangle(tlr_mask, (i[0]-i[2]*4, i[1]-i[2]*8), (i[0]+i[2]*4, i[1]+i[2]*8),(255), -1)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),1,(0,0,255),1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
img_tophat = cv2.morphologyEx(img_grayscale, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Circles", img)
#cv2.imshow("TLRMask",tlr_mask)
cv2.waitKey()
cv2.destroyWindow("Circles")
#cv2.destroyWindow("TLRMask")


for roi in ROIs:
    cv2.imshow("ROI", roi)
    cv2.waitKey()
    cv2.destroyWindow("ROI")

#img_filtered = cv2.bitwise_and(img, img, mask=tlr_mask)
img_grayscale = cv2.cvtColor(img_filtered , cv2.COLOR_BGR2GRAY)


img_edges = cv2.Canny(img_grayscale,50,150,apertureSize = 3)
cv2.imshow("Edge", img_edges)
cv2.waitKey()
lines = cv2.HoughLinesP(img_edges,1,np.pi/180, 30, minLineLength=15, maxLineGap=20 )
print("lines:\n")
print(lines)
if lines is not None:
    for line in lines: 
        print(line)
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1) 

cv2.imshow("Lines",img)

#img_adaptiveThresholdMean = cv2.adaptiveThreshold(img_tophat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
#img_adaptiveThresholdGauss = cv2.adaptiveThreshold(img_tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
#cv2.imshow("MeanThreshold", img_adaptiveThresholdMean)
#cv2.imshow("GaussThreshold", img_adaptiveThresholdGauss)
#cv2.imshow("Slika", img)

cv2.waitKey()
cv2.destroyAllWindows()
cv2.waitKey()