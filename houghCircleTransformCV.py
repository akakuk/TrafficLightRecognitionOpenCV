import numpy as np
import cv2
import time

start = time.perf_counter()

imgOrg = cv2.imread("Circles.png")
img = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(img, cv2.CV_32F , 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_32F , 0, 1, ksize=5)
mag, ang = cv2.cartToPolar(sobelx, sobely)
div = np.divide(sobely, sobelx)
normals = np.tan(div)