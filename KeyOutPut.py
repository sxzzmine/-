import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from SIFT_FileKey import SIFT_frameKeyP
from SIFT_FileKey import SIFT_fileKeyP
from SURF_FileKey import SURF_frameKeyP
from SURF_FileKey import SURF_fileKeyP
from SURF_FileKey import SURF_Allfile
from Remove import remove_overlapping_Kd
from OutputTime import OutputT


def SIFT_keypoints(video_file, dpi=100):
    start_time = time.time()

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    if not os.path.exists('KeyOutput/SIFT'):
        os.makedirs('KeyOutput/SIFT')

    # 读取前十帧
    for i in range(100):
        ret, frame = video.read()

        if not ret:
            break

        # 检测SIFT关键点
        keypoints = sift.detect(frame, None)

        #keypoints, descriptors = sift.detectAndCompute(frame, None)

        # 在图像上绘制关键点
        #img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        #plt.figure(figsize=(10, 5),dpi = dpi)

        #plt.subplot(1, 2, 1)
        #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #plt.title('Original Image')

        #plt.subplot(1, 2, 2)
        #plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        #plt.title('Image with SIFT keypoints')

        #plt.savefig(f'KeyOutput/SIFT/SIFT_frame_{i}.png')
        #plt.show()

    OutputT(start_time)

    # 释放视频对象
    video.release()

def Compare_keypoints(video_file, dpi=100):#保留五百个最好的点

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    if not os.path.exists('KeyOutput/BestSIFT'):
        os.makedirs('KeyOutput/BestSIFT')

    # 读取前十帧
    for i in range(10):
        ret, frame = video.read()

        if not ret:
            break

        # 检测SIFT关键点
        keypoints = sift.detect(frame, None)
        TkeyP = SIFT_frameKeyP(i)
        TkeyP.computeDescriptors(frame)
        TkeyP.retainBest(500, frame)
        # 在图像上绘制关键点
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)
        img_with_best_keypoints = cv2.drawKeypoints(frame, TkeyP.getKeypoints(), None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        plt.figure(figsize=(15, 5),dpi = dpi)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with SIFT keypoints')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_with_best_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with Bset SIFT keypoints')



        plt.savefig(f'KeyOutput/BestSIFT/SIFT_frame_{i}.png')
        #plt.show()

    # 释放视频对象
    video.release()

def Compare_Remove_keypoints(video_file, dpi=100):#去除重复后保留500个最好的点

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    if not os.path.exists('KeyOutput/RemoveAndBestSIFT'):
        os.makedirs('KeyOutput/RemoveAndBestSIFT')


    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频文件{video_file}的帧数为{frame_count}')
    Filekey1 = SIFT_fileKeyP(0, frame_count)
    for i in range(10):
        ret, frame = video.read()
        if not ret:
            print("无法读取视频帧")
            break
        TkeyP = SIFT_frameKeyP(i)
        TkeyP.computeDescriptors(frame)
        TkeyP.retainBest(1000, frame)
        #TkeyP.retainRan(1000, frame)
        Filekey1.set_element(i, TkeyP)
    video.release()

    video = cv2.VideoCapture(video_file)
    # 读取前十帧
    for i in range(10):
        ret, frame = video.read()

        if not ret:
            break

        keypoints = sift.detect(frame, None)
        # 检测SIFT关键点
        TkeyP1 =Filekey1.frames[i]
        TkeyP2 = Filekey1.frames[i + 1]
        MykeyP1 = TkeyP1.keypoints
        Mydescriptors1 = TkeyP1.descriptors
        MykeyP2 = TkeyP2.keypoints
        Mydescriptors2 = TkeyP2.descriptors

        TkeyP1.keypoints, TkeyP1.descriptors = remove_overlapping_Kd(MykeyP1, MykeyP2, threshold=2)

        # 在图像上绘制关键点
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)
        img_with_best_keypoints = cv2.drawKeypoints(frame, TkeyP1.getKeypoints(), None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        plt.figure(figsize=(15, 5),dpi = dpi)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with SIFT keypoints')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_with_best_keypoints, cv2.COLOR_BGR2RGB))
        #plt.title('Image with RemoveAndBset')
        plt.title('Image with RemoveAndRands')


        plt.savefig(f'KeyOutput/RemoveAndBestSIFT/SIFT_frame_{i}.png')
        #plt.savefig(f'KeyOutput/RemoveAndRandSIFT/SIFT_frame_{i}.png')
        #plt.show()

    # 释放视频对象
    video.release()



def Harris_keypoints(video_file, dpi=100):
    # 创建输出目录
    if not os.path.exists('KeyOutput/Harris'):
        os.makedirs('KeyOutput/Harris')

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 读取前十帧
    for i in range(10):
        ret, frame = video.read()

        if not ret:
            break

        # 转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Harris角点检测
        dst = cv2.cornerHarris(gray_frame, 2, 1, 0.04)

        result = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        image_with_keypoints = frame.copy()
        # 将角点位置标记为红色
        image_with_keypoints[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 使用matplotlib显示原始图像和带有关键点的图像
        plt.figure(figsize=(10, 5),dpi = dpi)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with Harris keypoints')

        plt.savefig(f'KeyOutput/Harris/Harris_frame_{i}.png')

        plt.show()



    # 释放视频对象
    video.release()

def ORB_keypoints(video_file, dpi=100):

    # 创建输出目录
    if not os.path.exists('KeyOutput/ORB'):
        os.makedirs('KeyOutput/ORB')

    start_time = time.time()
    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建ORB特征检测器
    #orb = cv2.ORB_create()
    orb = cv2.ORB_create(nfeatures=10000)
    # 读取前十帧
    for i in range(10):
        ret, frame = video.read()

        if not ret:
            break

        # 检测ORB关键点
        keypoints = orb.detect(frame, None)
        #keypoints, descriptors = orb.detectAndCompute(frame, None)

        # 在图像上绘制关键点
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        plt.figure(figsize=(10, 5), dpi=dpi)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with ORB keypoints')

        plt.savefig(f'KeyOutput/ORB/ORB_frame_{i}.png')

        plt.show()

    OutputT(start_time)
    # 释放视频对象
    video.release()




def SURF_keypoints(video_file, dpi=100):

    start_time = time.time()
    # 创建输出目录
    if not os.path.exists('KeyOutput/SURF'):
        os.makedirs('KeyOutput/SURF')

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建SURF特征检测器
    #sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(500)

    # 读取前十帧
    for i in range(100):
        ret, frame = video.read()

        if not ret:
            break

        # 检测SURF关键点

        keypoints = surf.detect(frame, None)
        #keypoints, descriptors = surf.detectAndCompute(frame, None)

        # 在图像上绘制关键点
        #img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        #plt.figure(figsize=(10, 5), dpi=dpi)

        #plt.subplot(1, 2, 1)
        #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #plt.title('Original Image')

        #plt.subplot(1, 2, 2)
        #plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        #plt.title('Image with SURF keypoints')

        #plt.savefig(f'KeyOutput/SURF/SURF_frame_{i}.png')

        #plt.show()

    OutputT(start_time)
    # 释放视频对象
    video.release()


def Compare_Remove_SURF(video_file, dpi=100):#去除重复后保留500个最好的点

    # 读取视频文件
    video = cv2.VideoCapture(video_file)

    # 创建SIFT特征检测器
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(500)

    if not os.path.exists('KeyOutput/RemoveAndBestSURF'):
        os.makedirs('KeyOutput/RemoveAndBestSURF')


    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频文件{video_file}的帧数为{frame_count}')
    FilekeyCR = SURF_fileKeyP(0, frame_count)
    for i in range(11):
        ret, frame = video.read()
        if not ret:
            print("无法读取视频帧")
            break
        TkeyP = SURF_frameKeyP(i)
        TkeyP.computeDescriptors(frame)
        TkeyP.retainBest(1000, frame)
        #TkeyP.retainRan(1000, frame)
        FilekeyCR.set_element(i, TkeyP)
    video.release()

    video = cv2.VideoCapture(video_file)
    # 读取前十帧
    for i in range(10):
        ret, frame = video.read()

        if not ret:
            break

        keypoints = surf.detect(frame, None)
        # 检测SIFT关键点
        MykeyP1 = FilekeyCR.frames[i].keypoints
        MykeyP2 = FilekeyCR.frames[i + 1].keypoints

        FilekeyCR.frames[i].keypoints  = remove_overlapping_Kd(MykeyP1, MykeyP2, threshold=2)

        FilekeyCR.frames[i].keypoints = sorted(FilekeyCR.frames[i].keypoints, key=lambda x: x.response, reverse=True)
        FilekeyCR.frames[i].keypoints = FilekeyCR.frames[i].keypoints[:200]

        # 在图像上绘制关键点
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)
        img_with_best_keypoints = cv2.drawKeypoints(frame, FilekeyCR.frames[i].getKeypoints(), None)

        # 使用matplotlib显示原始图像和带有关键点的图像
        plt.figure(figsize=(15, 5),dpi = dpi)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with SURF keypoints')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_with_best_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with RemoveAndBest')
        #plt.title('Image with RemoveAndRands')


        plt.savefig(f'KeyOutput/RemoveAndBestSURF/SURF_frame_{i}.png')
        #plt.savefig(f'KeyOutput/RemoveAndRandSURF/SURF_frame_{i}.png')
        plt.show()

    # 释放视频对象
    video.release()