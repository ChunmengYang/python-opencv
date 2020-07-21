#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


time.sleep(60)

threshold = 30
learningRate = 0.8
background = None
backImage = None
foreground = None

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

# 获取视频的宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 获取视频的高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

out_win = 'Dynamic Capture'
cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(1):
    _, frame = cap.read()
    # 高斯模糊
    # frame = cv2.GaussianBlur(frame,(3, 3), 0)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if background is None:
        background = np.float32(frame_gray)
    backImage = cv2.convertScaleAbs(background)

    foreground = cv2.absdiff(frame_gray, backImage)
    retval, foreground = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY_INV)
     # 膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # foreground = cv2.dilate(foreground, kernel, iterations=1)

    cv2.accumulateWeighted(frame_gray, background, learningRate)


    # _, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     if cv2.contourArea(c) < 500:
    #         continue
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('frame', frame)

    cv2.imshow(out_win, foreground)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()