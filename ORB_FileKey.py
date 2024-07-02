#'

import numpy as np
import pickle
import random

import cv2


from Remove import remove_overlapping_Kd
from Remove import remove_overlapping




orb = cv2.ORB_create(nfeatures= 8000)


class ORB_frameKeyP:
    def __init__(self,id):
        self.fid = id
        self.keypoints = []
        self.descriptors = []

    def retainBest(self, num,frame):
        # 从关键点列表中选择最好的num个关键点
        # 按照关键点的响应值进行排序
        keypoints = sorted(self.keypoints, key=lambda x: x.response, reverse=True)
        self.keypoints = keypoints[:num]
        self.keypoints,self.descriptors = orb.compute(frame, self.keypoints)

    def retainRan(self, num,frame):

        self.keypoints = random.sample(self.keypoints, num)
        self.keypoints, self.descriptors = orb.compute(frame, self.keypoints)

    def computeKey(self, frame):
        self.keypoints = orb.detect(frame, None)

    def computeDescriptors(self, frame):
        # 计算关键点的描述符
        self.keypoints, self.descriptors = orb.detectAndCompute(frame, None)

    def getKeypoints(self):
        return self.keypoints

    def getDescriptors(self):
        return self.descriptors





class ORB_fileKeyP:
    def __init__(self, filenum,num_elements):
        self.filenum = filenum
        self.frames = [ORB_frameKeyP(i) for i in range(num_elements)]

    def get_element(self, index):
        return self.frames[index]

    def set_element(self, index, element):
        self.frames[index] = element


    def get_keypoints(self):
        return [element.getKeypoints() for element in self.frames]

    def get_descriptors(self):
        return [element.getDescriptors() for element in self.frames]

    def read_video_ctrl(self, video_path, frame_count, point_count):
        video = cv2.VideoCapture(video_path)
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("无法读取视频帧")
                break
            self.frames[i].computeKey(frame)
            #self.frames[i].computeDescriptors(frame)
            self.frames[i].keypoints = sorted(self.frames[i].keypoints, key=lambda x: x.response, reverse=True)
            self.frames[i].keypoints = self.frames[i].keypoints[:point_count]
        video.release()

    def remove_overlapping_keypoints(self, video_path, frame_count, point_count):
        video = cv2.VideoCapture(video_path)
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("无法读取视频帧")
                break
            self.frames[i].keypoints = remove_overlapping_Kd(self.frames[i].keypoints, self.frames[i + 1].keypoints, threshold=2)
            #self.frames[i].keypoints = remove_overlapping(self.frames[i].keypoints, self.frames[i + 1].keypoints, threshold=2)
            self.frames[i].retainBest(point_count, frame)
        self.frames = self.frames[:frame_count]
        video.release()




class ORB_Allfile:
    def __init__(self,num_files):
        self.num_files = num_files
        self.files = [ORB_fileKeyP(i, 0) for i in range(num_files)]

    def get_element(self, index):
        return self.files[index]

    def set_element(self, index, element):
        self.files[index] = element

    def append_element(self, element):
        self.files.append(element)




