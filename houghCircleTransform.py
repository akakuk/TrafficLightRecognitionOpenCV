import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



start = time.perf_counter()


 
img = cv2.imread("25914.png")
#img = cv2.imread("houghCircle.png")
img = img[0:int(img.shape[0]/2), 0:int(img.shape[1]), 0:3].copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (5, 5))
img_non_filter = cv2.Canny(img_blur, 100, 200)


 


imgOrg = cv2.imread("imgThreshold.png")
img_filter = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
#img_filter = cv2.Canny(img_filter, 100, 200)

img_and = cv2.bitwise_and(img_filter, img_non_filter)

cv2.imshow("Filter", img_filter)
cv2.imshow("No filter", img_non_filter)
cv2.imshow("And", img_and)
cv2.waitKey()
cv2.destroyAllWindows()

img = img_filter
imgOrg = img
maxRad = 12
minRad = 2

acc = np.zeros((img.shape[1], img.shape[0], maxRad - minRad))
#print(img.shape[1])
for w in range(0, img.shape[1] - 1):
    for h in range(0, img.shape[0] - 1):
        #print((w + h*img.shape[1])/(img.shape[0]*img.shape[1]))
        if img[h, w] == 255:
            for r in range(0, maxRad - minRad):
                for th in range(0, 360):
                    a = int(round(w - (r + minRad) * np.cos((th * np.pi) / 180)))
                    b = int(round(h - (r + minRad) * np.sin((th * np.pi) / 180)))
                    if a >= 0 and a < img.shape[1] and b >= 0 and b < img.shape[0]:
                        acc[a, b, r] += 1
                        #print(acc[a, b, r])
circlePoints = []
isPart = 0
notPart = 0
#print(np.max(acc[a-1:a+1, b-1:b+1, r-1:r+1]))
end = time.perf_counter()
print(end - start)
acc=np.pad(acc, ((1,1),(1,1),(1,1)), mode="constant", constant_values=0)
for r  in range(acc.shape[2] - 1,  1, -1):
    for a in range(1, acc.shape[0] - 1):
        for b in range(1, acc.shape[1] - 1):
            if r + minRad - 1 > a:
                radSumA = a
            elif r + minRad - 1 > acc.shape[0] - a:
                radSumA = acc.shape[0] - a
            else:
                radSumA =  r + minRad -1
            if r + minRad - 1 > b:
                radSumB = b
            elif r + minRad - 1 > acc.shape[1] - b:
                radSumA = acc.shape[0] - b
            else:
                radSumB =  r + minRad - 1
            if acc[a, b, r] > 150 and np.max(acc[a-radSumA:a+radSumA, b-radSumB:b+radSumB, r-1:r+1]) <= acc[a, b, r]:
                for th in range(0, 360):
                    x = int(round(a - (r + minRad - 1) * np.cos((th * np.pi) / 180)))
                    y = int(round(b - (r + minRad - 1) * np.sin((th * np.pi) / 180)))
                    if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                        if (x, y) not in circlePoints:
                            circlePoints.append((x, y))
                            if img[y, x] == 255:
                                isPart = isPart + 1
                            else:
                                notPart = notPart + 1
                if isPart / (isPart + notPart) >= 0.0:
                    cv2.circle(imgOrg, (a - 1, b - 1), r + minRad - 1, (0, 0, 255), 1)
                    acc[a-radSumA:a+radSumA, b-radSumB:b+radSumB, 0:r] = 0
                circlePoints.clear()
                isPart = 0
                notPart = 0
end = time.perf_counter()
print(end - start)
cv2.imshow("img", imgOrg)
cv2.imwrite("houghCircle.png", imgOrg)
cv2.waitKey()
cv2.destroyAllWindows()
lum_img = acc[:, :, 5]
plt.figure(figsize = (25, 40))
plt.imshow(lum_img, cmap="hot")
plt.colorbar()










