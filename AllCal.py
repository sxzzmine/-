
import cv2
import os
import time
import re

from Sortfile import Sortexten

from ORB_FileKey import ORB_fileKeyP
from ORB_FileKey import ORB_frameKeyP
from ORB_FileKey import ORB_Allfile
from SIFT_FileKey import SIFT_fileKeyP
from SIFT_FileKey import SIFT_Allfile
from SURF_FileKey import SURF_fileKeyP
from SURF_FileKey import SURF_Allfile

from Calcu_Match import calculate_differences_onebyone_F
from Calcu_Match import calculate_differences_onebyone_O
from Calcu_Match import calculate_differences_threeone_F
from Calcu_Match import calculate_differences_threeone_O


Select_file = 2





def CalORB(path, Select_frame, Frame_Windows, Input_frame, First_point, Final_point,choiceMode):
    # 使用示例
    start_time = time.time()
    last_time = time.time()
    now_time = time.time()

    input_path = path

    nums_file, filelist = Sortexten(input_path, ".mp4")

    Allmp4 = ORB_Allfile(nums_file)

    #Allmp4 = SURF_Allfile(Select_File)

    for i in range(nums_file):
    #for i in range(Select_file):
        file = filelist[i]
        pattern = re.compile(r'\d+')
        result = pattern.findall(file)
        filename = result[0]

        input_file_path = os.path.join(input_path, file)
        video = cv2.VideoCapture(input_file_path)
        Fileframe_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'视频文件{input_file_path}的帧数为{Fileframe_count}')
        video.release()

        FileKeyi = ORB_fileKeyP(filename, Input_frame + 1)

        FileKeyi.read_video_ctrl(input_file_path, Input_frame + 1, First_point)

        now_time = time.time()
        print(f'提取视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()

        FileKeyi.remove_overlapping_keypoints(input_file_path, Input_frame, Final_point)

        now_time = time.time()
        print(f'筛选视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()


        Allmp4.set_element(i, FileKeyi)
    if choiceMode == 2:
        offsetone, offsetall = calculate_differences_threeone_O(Allmp4, Frame_Windows, Input_frame)
    else:
        offsetone, offsetall = calculate_differences_onebyone_O(Allmp4, Frame_Windows, Input_frame)

    end_time = time.time()
    print(f'执行全部代码耗时为{end_time - start_time}秒')

    print("offsetone:", offsetone)

    print("offsetall:", offsetall)

    return offsetall







def CalSIFT(path, Select_frame, Frame_Windows, Input_frame, First_point, Final_point, choiceMode):
    # 使用示例
    start_time = time.time()
    last_time = time.time()
    now_time = time.time()

    input_path = path

    nums_file, filelist = Sortexten(input_path, ".mp4")

    Allmp4 = SIFT_Allfile(nums_file)

    #Allmp4 = SURF_Allfile(Select_File)

    for i in range(nums_file):
    #for i in range(Select_file):
        file = filelist[i]
        pattern = re.compile(r'\d+')
        result = pattern.findall(file)
        filename = result[0]

        input_file_path = os.path.join(input_path, file)
        video = cv2.VideoCapture(input_file_path)
        Fileframe_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'视频文件{input_file_path}的帧数为{Fileframe_count}')
        video.release()

        FileKeyi =SIFT_fileKeyP(filename, Input_frame + 1)

        FileKeyi.read_video_ctrl(input_file_path, Input_frame + 1, First_point)

        now_time = time.time()
        print(f'提取视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()

        FileKeyi.remove_overlapping_keypoints(input_file_path, Input_frame, Final_point)

        now_time = time.time()
        print(f'筛选视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()


        Allmp4.set_element(i, FileKeyi)

    if choiceMode == 2:
        offsetone, offsetall = calculate_differences_threeone_F(Allmp4, Frame_Windows, Input_frame)
    else:
        offsetone, offsetall = calculate_differences_onebyone_F(Allmp4, Frame_Windows, Input_frame)

    end_time = time.time()
    print(f'执行全部代码耗时为{end_time - start_time}秒')

    print("offsetone:", offsetone)

    print("offsetall:", offsetall)

    return offsetall




def CalSURF(path, Select_frame, Frame_Windows, Input_frame, First_point, Final_point, choiceMode):
    # 使用示例
    start_time = time.time()
    last_time = time.time()
    now_time = time.time()

    input_path = path

    nums_file, filelist = Sortexten(input_path, ".mp4")

    Allmp4 = SURF_Allfile(nums_file)

    #Allmp4 = SURF_Allfile(Select_File)

    for i in range(nums_file):
        file = filelist[i]
        pattern = re.compile(r'\d+')
        result = pattern.findall(file)
        filename = result[0]

        input_file_path = os.path.join(input_path, file)
        video = cv2.VideoCapture(input_file_path)
        Fileframe_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'视频文件{input_file_path}的帧数为{Fileframe_count}')
        video.release()

        FileKeyi = SURF_fileKeyP(filename, Input_frame + 1)

        FileKeyi.read_video_ctrl(input_file_path, Input_frame + 1, First_point)

        now_time = time.time()
        print(f'提取视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()

        FileKeyi.remove_overlapping_keypoints(input_file_path, Input_frame, Final_point)

        now_time = time.time()
        print(f'筛选视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
        last_time = time.time()


        Allmp4.set_element(i, FileKeyi)
    if choiceMode == 2:
        offsetone, offsetall = calculate_differences_threeone_F(Allmp4, Frame_Windows, Input_frame)
    else:
        offsetone, offsetall = calculate_differences_onebyone_F(Allmp4, Frame_Windows, Input_frame)

    end_time = time.time()
    print(f'执行全部代码耗时为{end_time - start_time}秒')

    print("offsetone:", offsetone)

    print("offsetall:", offsetall)

    return offsetall