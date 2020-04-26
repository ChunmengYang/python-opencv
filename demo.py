#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图片
# cv2.read(imagefile, parms)对于第二个参数，可以使用-1，0或1。
# 颜色为1，灰度为0，不变为-1。因此，对于灰度，可以执行cv2.imread('watch.jpg', 0)。
img = cv2.imread('glnz2.jpeg', 1)
# BGR转灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图转RGB
img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
# BGR转RGB
img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 保存图片
cv2.imwrite('glnz_gray.jpeg', img_gray)

# # img.shape可以获取图像的形状。他的返回值是一个包含行数，列数，通道数的元组。
# print(img.shape)
# # img.size返回图像的像素数目
# print(img.size)
# #img.dtype返回的是图像的数据类型.
# print(img.dtype)



# # 绘制线条, cv2.line()接受以下参数：图片，开始坐标，结束坐标，颜色（bgr），线条粗细。
# cv2.line(img, (0,0), (150,150), (255,0,0), 5)

# # 绘制矩形, cv2.rectangle()接受以下参数：图像，左上角坐标，右下角坐标，颜色（bgr），线条粗细。
# cv2.rectangle(img,(0,0), (150,150), (0,0,255), 5)

# # 绘制圆形, cv2.circle()接受以下参数：图像，圆心，半径，颜色（bgr)，线条粗细。 注意我们粗细为-1。 这意味着将填充对象，所以我们会得到一个圆。
# cv2.circle(img, (100,100), 55, (0,255,0), -1)

# # 绘制椭圆, cv2.ellipse()接受以下参数：图片，椭圆中心，x/y轴的长度，椭圆的旋转角度，椭圆的起始角度，椭圆的结束角度，颜色，字体粗细。注意我们粗细为-1，这意味着将填充对象。
# cv2.ellipse(img, (200, 200), (100, 50), 0, 0, 180, (255, 0, 0), -1)

# # 定义四个顶点坐标
# pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
# pts = pts.reshape((-1,1,2))
# # 绘制多边形, cv2.polylines()接受以下参数：图像，多个顶级，多边行是否闭合，颜色（bgr），线条粗细。
# cv2.polylines(img, [pts], True, (0,255,255), 3)

# # 绘制文字， cv2.putText()接受以下参数：图片，文字，左上角坐标，字体，字体大小，颜色，字体粗细
# cv2.putText(img,'OpenCV Tuts!', (10,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,255,155), 2, cv2.LINE_AA)

# cv2.imshow('Image', img)




# # 阈值的思想是进一步简化视觉数据的分析。
# # cv2.threshold()接受以下参数：图片，阈值（起始值），最大值，使用的是什么类型的算法，常用值为0（cv2.THRESH_BINARY）。
# # 首先，你可以转换为灰度，但是你必须考虑灰度仍然有至少 255 个值。
# # 阈值可以做的事情，在最基本的层面上，是基于阈值将所有东西都转换成白色或黑色。
# # 比方说，我们希望阈值为 125（最大为 255），那么 125 以下的所有内容都将被转换为 0 或黑色，而高于 125 的所有内容都将被转换为 255 或白色。
# # 如果你像平常一样转换成灰度，你会变成白色和黑色。如果你不转换灰度，你会得到二值化的图片，但会有颜色。
# # cv2.THRESH_BINARY
# # 大于阈值的像素点的灰度值设定为最大值(如8位灰度值最大为255)，灰度值小于阈值的像素点的灰度值设定为0。
# # cv2.THRESH_BINARY_INV
# # 大于阈值的像素点的灰度值设定为0，而小于该阈值的设定为255。
# # cv2.THRESH_TRUNC
# # 像素点的灰度值小于阈值不改变，大于阈值的灰度值的像素点就设定为该阈值。
# # cv2.THRESH_TOZERO
# # 像素点的灰度值小于该阈值的不进行任何改变，而大于该阈值的部分，其灰度值全部变为0。
# # cv2.THRESH_TOZERO_INV 
# # 像素点的灰度值大于该阈值的不进行任何改变，像素点的灰度值小于该阈值的，其灰度值全部变为0。
# retval, threshold = cv2.threshold(img_gray, 80, 255, cv2.THRESH_TOZERO)

# # 自适应阈值二值化函数根据图片一小块区域的值来计算对应区域的阈值，从而得到也许更为合适的图片。
# # 当同一幅图像上的不同部分的具有不同亮度时。这种情况下我们需要采用自适应阈值。
# # 此时的阈值是根据图像上的每一个小区域计算与其对应的阈值。
# # 因此在同一幅图像上的不同区域采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。
# # cv2.adaptiveThreshold()
# # 参数1：要处理的原图
# # 参数2：当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值，一般为255
# # 参数3：小区域阈值的计算方式
# # ADAPTIVE_THRESH_MEAN_C：小区域内取均值
# # ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
# # 参数4：阈值方式（跟前面讲的那5种相同）
# # 参数5：小区域的面积，如11就是11*11的小块
# # 参数6：最终阈值等于小区域计算出的阈值再减去此值
# adaptiveThreshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

# cv2.imshow('Original', threshold)
# cv2.imshow('Adaptive threshold', adaptiveThreshold)



# # cv2.inRange函数设阈值，去除背景部分
# # mask = cv2.inRange(hsv, lower_red, upper_red)
# # hsv 指的是原图
# # lower_red 指的是图像中低于这个lower_red的值，图像值变为0
# # upper_red 指的是图像中高于这个upper_red的值，图像值变为0
# # 而在lower_red～upper_red之间的值变成255
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_red = np.array([0, 0, 0])
# upper_red = np.array([180, 255, 65]) 
# mask = cv2.inRange(hsv, lower_red, upper_red)
# # 图像位运与算，是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
# res = cv2.bitwise_and(img, img, mask= mask)



# # 图片缩放
# imgscal = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
# cv2.imshow('image', imgscal)
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)



# # 2D卷积
# kernel = np.ones((5, 5), np.float32) / 25 
# smoothed = cv2.filter2D(img, -1, kernel) 
# # 平均模糊
# blur = cv2.blur(img, (5, 5))
# # 高斯模糊
# blur = cv2.GaussianBlur(img,(5, 5), 0)
# # 中值模糊
# blur = cv2.medianBlur(img, 5)
# 双向模糊
# blur = cv2.bilateralFilter(img,9,75,75)

# cv2.imshow('Original', img) 
# cv2.imshow('Averaging', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



 
# 边缘检测cv2.Canny()
# 第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
# 第二个参数是阈值1；
# 第三个参数是阈值2。
# 其中较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
# cv2.imshow('Original', img)
# edges = cv2.Canny(img, 100, 200) 
# cv2.imshow('Edges', edges)




# # 边缘计算
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

# absX = cv2.convertScaleAbs(sobelx)
# absY = cv2.convertScaleAbs(sobely)
# absXY = cv2.convertScaleAbs(sobelxy)
# dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

# cv2.imshow('Original', img)
# cv2.imshow('Sobelxy', dst)



# template = cv2.imread('face.jpg', 0)
# w, h = template.shape[::-1]

# # 最大匹配值的坐标
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# left_top = max_loc  # 左上角
# right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
# cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置

# # 匹配多个物体
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.9
# # 匹配程度大于%80的坐标y,x
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):  # *号表示可选参数
#     right_bottom = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)
# cv2.imshow('Match Template', img)



# GrabCut 前景矩形提取
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1,65), np.float64)
# fgdModel = np.zeros((1,65), np.float64)
# rect = (1, 1, img.shape[1], img.shape[0])

# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img * mask[:,:,np.newaxis]
# cv2.imshow('Grab Cut GC_INIT_WITH_RECT', img) 

# # GrabCut 前景掩图提取
# newmask = cv2.imread('glnz2_mark.png', 0)
# mask[newmask == 0] = 0
# mask[newmask == 255] = 1

# mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img * mask[:,:,np.newaxis]
# cv2.imshow('Grab Cut GC_INIT_WITH_MASK', img)



# canny = cv2.Canny(img_gray, 80, 150)
# cv2.imshow('Canny', canny)
# #寻找图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
# canny,contours,hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# #32位有符号整数类型，
# marks = np.zeros(img.shape[:2],np.int32)
# #findContours检测到的轮廓
# imageContours = np.zeros(img.shape[:2],np.uint8)

# #轮廓颜色
# compCount = 0
# index = 0
# #绘制每一个轮廓
# for index in range(len(contours)):
#     #对marks进行标记，对不同区域的轮廓使用不同的亮度绘制，相当于设置注水点，有多少个轮廓，就有多少个轮廓
#     #图像上不同线条的灰度值是不同的，底部略暗，越往上灰度越高
#     marks = cv2.drawContours(marks,contours,index,(index,index,index),1,8,hierarchy)
#     #绘制轮廓，亮度一样
#     imageContours = cv2.drawContours(imageContours,contours,index,(255,255,255),1,8,hierarchy)


# markerShows = cv2.convertScaleAbs(marks)    
# cv2.imshow('markerShows', markerShows)


# marks = cv2.watershed(img, marks)
# afterWatershed = cv2.convertScaleAbs(marks)  
# cv2.imshow('afterWatershed', afterWatershed)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # Opencv 中的函数 cv2.cornerHarris() 可以用来进行角点检测。参数如下：
# # 　　• img - 数据类型为 float32 的输入图像。
# # 　　• blockSize - 角点检测中要考虑的领域大小。
# # 　　• ksize - Sobel 求导中使用的窗口大小
# # 　　• k - Harris 角点检测方程中的自由参数，取值参数为 [0,04，0.06].
# img_gray = np.float32(img_gray)
# dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()] = [0,0,255]

# # cv2.goodFeaturesToTrack()
# # 第一个参数是输入图像。
# # 第二个参数用于限定检测到的点数的最大值。
# # 第三个参数表示检测到的角点的质量水平（通常是0到1之间的数值，不能大于1.0）。
# # 第四个参数用于区分相邻两个角点的最小距离（小于这个距离得点将进行合并）。
# corners = cv2.goodFeaturesToTrack(img_gray, 50, 0.5, 10)
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# cv2.imshow('dst', img)



# # Initiate ORB detector
# orb = cv2.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(img, None)
# # compute the dscriptors with ORB
# kp, des = orb.compute(img, kp)

# # draw only keypoints location, not size and orientation
# img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
# cv2.imshow('Keypoints', img_kp) 


# img_face = cv2.imread('face.jpg', 1)
# orb = cv2.ORB_create()
# img_kp, img_des = orb.detectAndCompute(img, None)
# face_kp, face_des = orb.detectAndCompute(img_face, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(face_des, img_des)
# matches = sorted(matches, key = lambda x:x.distance)
# img_matches = cv2.drawMatches(img_face, face_kp, img, img_kp, matches[:10], None, flags=2)
# cv2.imshow('Keypoints Matches', img_matches) 



# frame = cv2.imread('5-2.jpg', 1)
# # BGR转灰度图
# frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 高斯滤波
# frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
# # 灰度图
# frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
# # 二值化
# retval, threshold = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_TOZERO)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# # # 形态开运算
# # frame_morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
# # cv2.imshow('frame_morph', frame_morph)
# # 腐蚀
# frame_erode = cv2.erode(threshold, kernel, iterations=1)
# cv2.imshow('frame_erode', frame_erode)
# # 膨胀
# frame_dilate = cv2.dilate(frame_erode, kernel, iterations=1)
# cv2.imshow('frame_dilate', frame_dilate)

# # 轮廓检测
# contours_map, contours, hierarchy = cv2.findContours(frame_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
# # cv2.drawContours(frame_dilate, contours, -1, ( 255,0,0), 1)  

# for cnt in contours:
#     # 凸包
#     hull = cv2.convexHull(cnt, returnPoints = True)
#     length = len(hull)
    
#     # 绘制图像凸包的轮廓
#     for i in range(length):
#         cv2.circle(frame, tuple(hull[i][0]), 2, [0,0,255], -1)
#         cv2.line(frame, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0, 255, 0), 1)

#     # 凸缺陷
#     hull = cv2.convexHull(cnt, returnPoints = False)
#     defects = cv2.convexityDefects(cnt, hull)
#     if not (defects is None):  
#         for i in range(defects.shape[0]):
#             s,e,f,d = defects[i,0]
#             start = tuple(cnt[s][0])
#             end = tuple(cnt[e][0])
#             far = tuple(cnt[f][0])
#             if d > 2000:
#                 cv2.circle(frame, far, 5, [255,0,0], -1)

# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# bg = cv2.imread('frame_bg.jpg', 1)
# bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
# retval, bg_threshold = cv2.threshold(bg_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# cv2.imshow('frame', bg_threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

threshold = 30
learningRate = 0.5
background = None
backImage = None
foreground = None

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)
# while(1):
#     _, frame = cap.read()
    
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if background is None:
#         background = np.float32(frame_gray)
#         continue

#     backImage = cv2.convertScaleAbs(background)
#     foreground = cv2.absdiff(frame_gray, backImage)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
#     foreground = cv2.dilate(foreground, kernel)

#     retval, foreground = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY_INV)

#     background = np.float32(frame_gray)
#     dst = cv2.resize(foreground, (40, 30))
#     cv2.imshow('foreground', dst)


#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()
# cap.release()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
while(1):
    _, frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if background is None:
        background = np.float32(frame_gray)

    backImage = cv2.convertScaleAbs(background)

    foreground = cv2.absdiff(frame_gray, backImage)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
    foreground = cv2.dilate(foreground, kernel)

    # retval, foreground = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.accumulateWeighted(frame_gray, background, learningRate)

    dst = cv2.resize(foreground, (40, 30))
    cv2.imshow('foreground', dst)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()