import cv2
import numpy as np
import re
import os
from Sortfile  import Sortexten

path = 'Input'



input_path = path

nums_file, filelist = Sortexten(input_path, ".mp4")

file = filelist[0]
pattern = re.compile(r'\d+')
result = pattern.findall(file)
filename = result[0]

input_file_path = os.path.join(input_path, file)
video = cv2.VideoCapture(input_file_path)

ret , frame1 = video.read()

video.release()

file = filelist[1]
pattern = re.compile(r'\d+')
result = pattern.findall(file)
filename = result[0]

input_file_path = os.path.join(input_path, file)
video = cv2.VideoCapture(input_file_path)

ret , frame2 = video.read()

video.release()


# 加载两张图片
img1 = frame1
img2 = frame2
# 初始化SIFT检测器
sift = cv2.xfeatures2d.SIFT_create()

# 查找关键点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# 使用FLANN进行特征匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# 仅存储良好匹配
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 计算本质矩阵
if len(good) > 10:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    E, mask = cv2.findEssentialMat(src_pts, dst_pts)

    # 计算相机位姿
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)


    # 使用计算得到的相机内参矩阵 mtx 和畸变系数 dist
    #E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # 计算相机位姿
    #_, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=mtx)

    # 计算夹角
    angle = np.arccos((np.trace(R) - 1) / 2)
    angle = np.degrees(angle)
    print('Angle: ', angle)

else:
    print("Not enough matches are found - %d/%d" % (len(good),10))



