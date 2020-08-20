import cPickle
from PyQt4 import QtCore, QtGui, uic
import numpy as n
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import os
import copy
from imutils.object_detection import non_max_suppression

frame = cv2.imread('example.png')
hog = cv2.HOGDescriptor()
# 设置支持向量机(Support Vector Machine)使得它成为一个预先训练好了的行人检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
rects = n.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

cv2.rectangle(frame, (pick[0,0], pick[0,1]), (pick[0,0] + pick[0,2], pick[0,1] + pick[0,3]), (0, 0, 255), 2)
cv2.imwrite('result.png',frame)