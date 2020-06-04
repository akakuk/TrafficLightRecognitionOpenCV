import numpy as np
import cv2

cap = cv2.VideoCapture("C:/Users/Kuka/Desktop/ffmpeg-4.2.2-win64-static/bin/Driving1.mp4")

while(cap.isOpened()):
    ret, img = cap.read()

    img = img[0:int(img.shape[0]/2), 0:int(img.shape[1]), 0:3].copy()
    
    img_blur = cv2.blur(img, (3, 3))
    
    #img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YUV)
    img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    #HSV filters
    lower_red = np.array([0,100,100])
    upper_red = np.array([20,255,255])
    lower_green = np.array([10,80,100])
    upper_green = np.array([30,255,255])
    
    
    #YUV filters
    #lower_red = np.array([120,85,140])
    #upper_red = np.array([255,120,170])
    #lower_green = np.array([120,20,24])
    #upper_green = np.array([225,127,96])
    
    #mask1 = cv2.inRange(img_yuv, lower_red, upper_red)
    mask1 = cv2.inRange(img_yuv, lower_green, upper_green)
    #mask = cv2.bitwise_or(mask1, mask2)
    
    res = cv2.bitwise_and(img_yuv, img_yuv, mask=mask1)
    
    cv2.imshow("Filter", res)
    
    circles = cv2.HoughCircles(mask1 , cv2.HOUGH_GRADIENT, 1, 100, param1=255, param2=8, minRadius=1, maxRadius=10)
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

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
