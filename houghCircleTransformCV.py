import numpy as np
import cv2
import time

start = time.perf_counter()

imgOrg = cv2.imread("Circles.png")
img = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(img, cv2.CV_32F , 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_32F , 0, 1, ksize=3)
mag, ang = cv2.cartToPolar(sobelx, sobely)
print(ang.shape)
acc = np.zeros((img.shape[1], img.shape[0]))
minRad = 2
maxRad = 10
for w in range(0, img.shape[1] - 1):
    for h in range(0, img.shape[0] - 1):
        if img[h,w] == 0:
            for r in range(minRad,maxRad):
                X = int(round(w + (r * np.cos(ang[h,w]))))
                Y = int(round(h + (r * np.sin(ang[h,w]))))
                if X >= 0 and X < img.shape[1] and Y >= 0 and Y < img.shape[0]:
                    acc[X,Y] = acc[X,Y] + 1
            
cv2.imshow("acc",acc)
cv2.waitKey()
print(np.where(acc == np.amax(acc)))