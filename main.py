from moviepy.editor import VideoFileClip
import cv2
import os
import time
import re


from SIFT_FileKey import SIFT_frameKeyP
from SIFT_FileKey import SIFT_fileKeyP
from SIFT_FileKey import SIFT_Allfile

from SURF_FileKey import SURF_frameKeyP
from SURF_FileKey import SURF_fileKeyP
from SURF_FileKey import SURF_Allfile

from Sortfile import Sortexten
from KeyOutPut import SIFT_keypoints
from KeyOutPut import Harris_keypoints
from KeyOutPut import ORB_keypoints
from KeyOutPut import Compare_keypoints
from KeyOutPut import Compare_Remove_keypoints
from KeyOutPut import SURF_keypoints
from KeyOutPut import Compare_Remove_SURF

from Remove import remove_overlapping_Kd

#from OutputTime import OutputT

from Calcu_Match import calculate_match_percentage_F
from Calcu_Match import output_match
from Calcu_Match import calculate_video_F
from Calcu_Match import find_max_sum_column
from Calcu_Match import calculate_differences_threeone_F
from Calcu_Match import calculate_differences_onebyone_F

Select_File = 7
#选择文件数


Select_frame = 150#匹配的有效帧数

Frame_Windows = 30#时域窗口大小

Input_frame = Select_frame + Frame_Windows * 2#匹配的读取帧数



First_point= 5000#提取特征点后根据响应度筛查后剩下的特征点

Final_point = 500#进行特征点筛选后剩下的有效特征点


start_time = time.time()
last_time = time.time()
now_time = time.time()

def get_video_info(video_file):
    clip = VideoFileClip(video_file)
    duration = clip.duration  # duration in seconds
    fps = clip.fps  # frames per second
    frames = clip.reader.nframes  # total frames
    video_size = clip.size  # size of video (width, height)
    audio_fps = clip.audio.fps  # audio frames per second
    return {
        "duration": duration,
        "fps": fps,
        "frames": frames,
        "video_size": video_size,
        "audio_fps": audio_fps,
    }



# 使用示例
input_path = 'Input'
txt_path = 'Trantxt'

nums_file ,filelist = Sortexten(input_path, ".mp4")


Allmp4 = SURF_Allfile(Select_File)
input_file = os.path.join(input_path, filelist[0])

#Compare_Remove_keypoints(input_file,400)
#Compare_keypoints(input_file,400)
#SIFT_keypoints(input_file,400)
#Harris_keypoints(input_file,400)
#ORB_keypoints(input_file,400)
#SURF_keypoints(input_file,400)
#Compare_Remove_SURF(input_file,400)


for i in range(Select_File):
    file = filelist[i]
    pattern = re.compile(r'\d+')
    result = pattern.findall(file)
    filename = result[0]

    input_file_path = os.path.join(input_path, file)
    video = cv2.VideoCapture(input_file_path)
    ret , frame = video.read()
    frame1 = frame
    ret, frame = video.read()
    frame2 = frame
    Fileframe_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频文件{input_file_path}的帧数为{Fileframe_count}')
    video.release()

    FileKeyi = SURF_fileKeyP(filename, Input_frame + 1)

    #FileKeyi = SIFT_fileKeyP(i, Select_frame + 1)

    FileKeyi.read_video_ctrl(input_file_path, Input_frame + 1, First_point)

    now_time = time.time()
    print(f'提取视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
    last_time = time.time()
    #OutputT(start_time)

    FileKeyi.remove_overlapping_keypoints(input_file_path, Input_frame, Final_point)

    now_time = time.time()
    print(f'筛选视频文件{input_file_path}的特征点耗时为{now_time - last_time}秒')
    last_time = time.time()
    #OutputT(start_time)

    #Match = calculate_match_percentage(FileKeyi.frames[0], FileKeyi.frames[1])

    #Match = output_match(frame1,frame2, FileKeyi.frames[0], FileKeyi.frames[1])


    Allmp4.set_element(i, FileKeyi)

#Video_Match = calculate_video(Allmp4.files[0], Allmp4.files[1], Frame_Windows, Input_frame)

#max_Matchs, max_match_index = find_max_sum_column(Video_Match)
#Best_frame_match = max_Matchs/Select_frame
#Best_frame_offset = max_match_index - Frame_Windows
#print("The maximum match percent is", Best_frame_match, "and the Best frame match is", Best_frame_offset)

#offset = calculate_differences_threeone(Allmp4, Frame_Windows, Input_frame)

offsetone,offsetall = calculate_differences_onebyone_F(Allmp4, Frame_Windows, Input_frame)


end_time = time.time()
print(f'执行全部代码耗时为{end_time- start_time}秒')
#OutputT(start_time)

#print("offset:", offset)

print("offsetone:", offsetone)

print("offsetall:", offsetall)

end_time = time.time()














#Allkey1.save_to_txt(txt_path)
#Allkey1.save_pkl(txt_path)
#Allkey2= fileKeyP(1,10)
#Allkey2.load_to_txt(txt_path)
