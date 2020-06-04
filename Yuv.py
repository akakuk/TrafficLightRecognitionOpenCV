import cv2
import numpy as np


img = cv2.imread("37596.png")
#img = cv2.imread("15096.png")
#img = cv2.imread("22008.png")
#img = cv2.imread("14888.png")
img = img[0:int(img.shape[0]/2), 0:int(img.shape[1]), 0:3].copy()
cv2.imshow("Slika", img)
cv2.waitKey()
cv2.destroyWindow("Slika")

img_blur = cv2.blur(img, (5, 5))

img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YUV_YV12)
cv2.imshow("SlikaYUV", img_yuv)
cv2.waitKey()
cv2.destroyWindow("SlikaYUV")

lower_red = np.array([90,20,129])
upper_red = np.array([225,126,190])
lower_green = np.array([90,20,24])
upper_green = np.array([225,127,96])

mask1 = cv2.inRange(img_yuv, lower_red, upper_red)
mask2 = cv2.inRange(img_yuv, lower_green, upper_green)
mask = cv2.bitwise_or(mask1, mask2)


res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Filter", res)
cv2.waitKey()
cv2.destroyWindow("Filter")

circles = cv2.HoughCircles(mask , cv2.HOUGH_GRADIENT, 1, 100, param1=255, param2=8, minRadius=1, maxRadius=20)
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
        
cv2.imshow("Slika", img)
cv2.waitKey()
cv2.destroyWindow("Slika")

cv2.waitKey()