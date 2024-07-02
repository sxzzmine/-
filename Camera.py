import numpy as np
import cv2
import glob

# 设置棋盘格的角点数
corner_x = 6
corner_y = 9

# 设置世界坐标下的棋盘格点的坐标
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)

# 用于存储所有图片的棋盘格角点和对应的世界坐标
objpoints = []
imgpoints = []

# 读取所有的棋盘格图片
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # 如果找到了足够的角点，将其添加到数据中
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 可视化角点
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 进行相机校准
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出内参矩阵和畸变系数
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)