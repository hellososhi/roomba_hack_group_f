import cv2
import numpy as np

img = cv2.imread('img.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_s = img_hsv[:, :, 1]
img_v = img_hsv[:, :, 2]
cv2.imwrite('img_s.png', img_s)
cv2.imwrite('img_v.png', img_v)
img_mask = cv2.inRange(img_hsv, (0, 100, 100), (255, 255, 255))
img_bin = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("img_bin.png", img_bin)
contours, hierarchy = cv2.findContours(255-img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imwrite('img_s_contours.png', img_contours)
cv2.imwrite('img_mask.png', img_mask)
