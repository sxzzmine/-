import cv2
import numpy as np
import numpy as np
from SURF_FileKey import SURF_frameKeyP
from SURF_FileKey import SURF_fileKeyP
from SURF_FileKey import SURF_Allfile

import time
from OutputTime import OutputT
def calculate_match_percentage_F(frameKeyP1, frameKeyP2):
    # 转换为灰度图像
    #img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SURF特征提取器
    #surf = cv2.xfeatures2d.SURF_create()

    # 提取关键点和特征描述符
    kp1 = frameKeyP1.keypoints
    des1 = frameKeyP1.descriptors
    kp2 = frameKeyP2.keypoints
    des2 = frameKeyP2.descriptors
    #kp1, des1 = surf.detectAndCompute(img1_gray, None)
    #kp2, des2 = surf.detectAndCompute(img2_gray, None)


    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用Lowe的比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # 计算匹配度
    match_percentage = len(good_matches) / len(matches) * 100

    return match_percentage


def calculate_match_percentage_O(frameKeyP1, frameKeyP2):
    # 提取关键点和特征描述符
    kp1 = frameKeyP1.keypoints
    des1 = frameKeyP1.descriptors
    kp2 = frameKeyP2.keypoints
    des2 = frameKeyP2.descriptors

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用Lowe的比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # 计算匹配度
    match_percentage = len(good_matches) / len(matches) * 100

    return match_percentage







def calculate_match_percentage_OO(frameKeyP1, frameKeyP2):
    # 提取关键点和特征描述符
    kp1 = frameKeyP1.keypoints
    des1 = frameKeyP1.descriptors
    kp2 = frameKeyP2.keypoints
    des2 = frameKeyP2.descriptors

    # 初始化BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行匹配
    matches = bf.match(des1, des2)

    # 根据距离进行排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用RANSAC方法估计单应性矩阵H
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 使用mask过滤错误匹配
    matches_mask = mask.ravel().tolist()

    # 计算匹配度
    match_percentage = sum(matches_mask) / len(matches_mask) * 100

    return match_percentage


def output_match(image1, image2, frameKeyP1, frameKeyP2):
    # 转换为灰度图像
    #img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SURF特征提取器
    #surf = cv2.xfeatures2d.SURF_create()

    # 提取关键点和特征描述符
    kp1 = frameKeyP1.keypoints
    des1 = frameKeyP1.descriptors
    kp2 = frameKeyP2.keypoints
    des2 = frameKeyP2.descriptors
    #kp1, des1 = surf.detectAndCompute(img1_gray, None)
    #kp2, des2 = surf.detectAndCompute(img2_gray, None)

    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用Lowe的比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    draw_good_matches(image1, image2, good_matches, kp1, kp2)
    # 计算匹配度
    match_percentage = len(good_matches) / len(matches) * 100

    return match_percentage

def draw_good_matches(image1, image2, good_matches, keypoints1, keypoints2):
    # 绘制匹配
    result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 调整图像大小，使其不超过3800*2000
    max_width, max_height = 3000, 1500
    height, width, _ = result.shape
    scale = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_result = cv2.resize(result, (new_width, new_height))

    # 保存并显示结果
    cv2.imwrite("good_matches.jpg", resized_result)
    cv2.imshow("Good Matches", resized_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def calculate_C(A, B, M):
    N = len(A)
    assert N == len(B), "A and B should have the same length."

    # 初始化一个空的数组 C
    C = np.zeros((N, 2 * M + 1))

    # 计算 C[a][k]
    for a in range(N):
        for k in range(2 * M + 1):
            if a - M + k >= 0 and a - M + k < N:  # 确保 B 的索引在有效范围内
                C[a][k] = A[a] + B[a - M + k]

    D = C[M:N -M,:]

    return C

def calculate_video_F(BasefileP, CalfileP, FrameWindows, Select_Frames):
    start_time = time.time()
    # 初始化一个空的数组 C
    C = np.zeros((Select_Frames, 2 * FrameWindows + 1))

    # 计算 C[a][k]
    for baseFrames in range(Select_Frames):
        for referFrames in range(2 * FrameWindows + 1):
            if baseFrames - FrameWindows + referFrames >= 0 and baseFrames - FrameWindows + referFrames < Select_Frames:  # 确保 B 的索引在有效范围内
                #C[a][k] = A[a] + B[a - FrameWindows + k]
                C[baseFrames][referFrames] = calculate_match_percentage_F(BasefileP.frames[baseFrames], CalfileP.frames[baseFrames - FrameWindows + referFrames])

    D = C[FrameWindows:Select_Frames - FrameWindows, :]
    OutputT(start_time)
    return D


def calculate_video_O(BasefileP, CalfileP, FrameWindows, Select_Frames):
    start_time = time.time()
    # 初始化一个空的数组 C
    C = np.zeros((Select_Frames, 2 * FrameWindows + 1))

    # 计算 C[a][k]
    for baseFrames in range(Select_Frames):
        for referFrames in range(2 * FrameWindows + 1):
            if baseFrames - FrameWindows + referFrames >= 0 and baseFrames - FrameWindows + referFrames < Select_Frames:  # 确保 B 的索引在有效范围内
                #C[a][k] = A[a] + B[a - FrameWindows + k]
                C[baseFrames][referFrames] = calculate_match_percentage_O(BasefileP.frames[baseFrames], CalfileP.frames[baseFrames - FrameWindows + referFrames])

    D = C[FrameWindows:Select_Frames - FrameWindows, :]
    OutputT(start_time)
    return D




def find_max_sum_column(matrix):
    # Convert the input matrix to a NumPy array
    matrix = np.array(matrix)

    # Calculate the sum of each column
    column_sums = np.sum(matrix, axis=0)

    # Find the index of the maximum sum
    max_column_index = np.argmax(column_sums)

    # Return the maximum sum and corresponding column index
    return column_sums[max_column_index], max_column_index

def calculate_differences_threeone_F(Allmp4, Frame_Windows, Input_frame):
    Filenum = len(Allmp4.files)
    Select_frame = Input_frame - Frame_Windows * 2
    FileR = Filenum % 3
    FileL= Filenum - FileR
    Filesets = FileL // 3

    OffsetOne =[0 for i in range(Filenum)]
    OffsetAll =[0 for i in range(Filenum)]

    for n in range(Filesets -1):
        Fnum_0 = Allmp4.files[3 * n + 1].filenum
        Fnum_1 = Allmp4.files[3 * n + 4].filenum
        Video_Match1 = calculate_video_F(Allmp4.files[3 * n + 1], Allmp4.files[3 * n + 4], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*n +4 ] = Best_frame_offset1
        OffsetAll[3*n +4 ] = Best_frame_offset1 + OffsetAll[3*n+1]
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0}]号文件帧偏移为 {Best_frame_offset1}")

    for n in range(Filesets):
        Fnum_0 = Allmp4.files[3 * n].filenum
        Fnum_1 = Allmp4.files[3 * n + 1].filenum
        Fnum_2 = Allmp4.files[3 * n + 2].filenum
        Video_Match1 = calculate_video_F(Allmp4.files[3 * n + 1], Allmp4.files[3 * n], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*n] = Best_frame_offset1
        OffsetAll[3*n] = Best_frame_offset1 + OffsetAll[3*n+1]

        Video_Match2 = calculate_video_F(Allmp4.files[3 * n + 1], Allmp4.files[3 * n + 2], Frame_Windows, Input_frame)
        max_Matchs2, max_match_index2 = find_max_sum_column(Video_Match2)
        Best_frame_match2 = max_Matchs2 / Select_frame
        Best_frame_offset2 = max_match_index2 - Frame_Windows
        OffsetOne[3*n+2] = Best_frame_offset2
        OffsetAll[3*n+2] = Best_frame_offset2 + OffsetAll[3*n+1]
        print(f"第[{Fnum_0}]号文件相比于第[{Fnum_1}]号文件帧偏移为 {Best_frame_offset1}，第[{Fnum_2}]号文件相比于第[{Fnum_1}]号文件帧偏移为{Best_frame_offset2}")

    for n in range(FileR):
        Fnum_0 = Allmp4.files[3*Filesets -2].filenum
        Fnum_1 = Allmp4.files[3*Filesets + n].filenum
        Video_Match1 = calculate_video_F(Allmp4.files[3*Filesets -2], Allmp4.files[3*Filesets + n], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*Filesets + n ] = Best_frame_offset1
        OffsetAll[3*Filesets + n ] = Best_frame_offset1 + OffsetAll[3*Filesets -2]
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0}]号文件帧偏移为 {Best_frame_offset1}")



    return OffsetOne,OffsetAll


def calculate_differences_threeone_O(Allmp4, Frame_Windows, Input_frame):
    Filenum = len(Allmp4.files)
    Select_frame = Input_frame - Frame_Windows * 2
    FileR = Filenum % 3
    FileL= Filenum - FileR
    Filesets = FileL // 3

    OffsetOne =[0 for i in range(Filenum)]
    OffsetAll =[0 for i in range(Filenum)]

    for n in range(Filesets -1):
        Fnum_0 = Allmp4.files[3 * n + 1].filenum
        Fnum_1 = Allmp4.files[3 * n + 4].filenum
        Video_Match1 = calculate_video_O(Allmp4.files[3 * n + 1], Allmp4.files[3 * n + 4], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*n +4 ] = Best_frame_offset1
        OffsetAll[3*n +4 ] = Best_frame_offset1 + OffsetAll[3*n+1]
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0}]号文件帧偏移为 {Best_frame_offset1}")

    for n in range(Filesets):
        Fnum_0 = Allmp4.files[3 * n].filenum
        Fnum_1 = Allmp4.files[3 * n + 1].filenum
        Fnum_2 = Allmp4.files[3 * n + 2].filenum
        Video_Match1 = calculate_video_O(Allmp4.files[3 * n + 1], Allmp4.files[3 * n], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*n] = Best_frame_offset1
        OffsetAll[3*n] = Best_frame_offset1 + OffsetAll[3*n+1]

        Video_Match2 = calculate_video_O(Allmp4.files[3 * n + 1], Allmp4.files[3 * n + 2], Frame_Windows, Input_frame)
        max_Matchs2, max_match_index2 = find_max_sum_column(Video_Match2)
        Best_frame_match2 = max_Matchs2 / Select_frame
        Best_frame_offset2 = max_match_index2 - Frame_Windows
        OffsetOne[3*n+2] = Best_frame_offset2
        OffsetAll[3*n+2] = Best_frame_offset2 + OffsetAll[3*n+1]
        print(f"第[{Fnum_0}]号文件相比于第[{Fnum_1}]号文件帧偏移为 {Best_frame_offset1}，第[{Fnum_2}]号文件相比于第[{Fnum_1}]号文件帧偏移为{Best_frame_offset2}")

    for n in range(FileR):
        Fnum_0 = Allmp4.files[3*Filesets -2].filenum
        Fnum_1 = Allmp4.files[3*Filesets + n].filenum
        Video_Match1 = calculate_video_O(Allmp4.files[3*Filesets -2], Allmp4.files[3*Filesets + n], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[3*Filesets + n ] = Best_frame_offset1
        OffsetAll[3*Filesets + n ] = Best_frame_offset1 + OffsetAll[3*Filesets -2]
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0}]号文件帧偏移为 {Best_frame_offset1}")



    return OffsetOne,OffsetAll


def calculate_differences_onebyone_O(Allmp4, Frame_Windows, Input_frame):
    Filenum = len(Allmp4.files)
    Select_frame = Input_frame - Frame_Windows * 2
    #if Filenum % 3 != 0:
    #    return "数组长度不是3的倍数"
    #Fileset = Filenum // 3
    OffsetOne =[0 for i in range(Filenum)]
    OffsetAll = [0 for i in range(Filenum)]
    Firstname = Allmp4.files[0].filenum
    for n in range(Filenum-1):
        Fnum_0 = Allmp4.files[n].filenum
        Fnum_1 = Allmp4.files[n+1].filenum
        Video_Match1 = calculate_video_O(Allmp4.files[n], Allmp4.files[n + 1], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        #Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[n+1] = Best_frame_offset1
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0 }]号文件帧偏移为 {Best_frame_offset1}")
        OffsetAll[n+1] = OffsetAll[n] + Best_frame_offset1
        print(f"第[{Fnum_1}]号文件相比于第[{Firstname}]号文件帧偏移为 {OffsetAll[n+1]}")

    return OffsetOne,OffsetAll


def calculate_differences_onebyone_F(Allmp4, Frame_Windows, Input_frame):
    Filenum = len(Allmp4.files)
    Select_frame = Input_frame - Frame_Windows * 2
    #if Filenum % 3 != 0:
    #    return "数组长度不是3的倍数"
    #Fileset = Filenum // 3
    OffsetOne =[0 for i in range(Filenum)]
    OffsetAll = [0 for i in range(Filenum)]
    Firstname = Allmp4.files[0].filenum
    for n in range(Filenum-1):
        Fnum_0 = Allmp4.files[n].filenum
        Fnum_1 = Allmp4.files[n+1].filenum
        Video_Match1 = calculate_video_F(Allmp4.files[n], Allmp4.files[n + 1], Frame_Windows, Input_frame)
        max_Matchs1, max_match_index1 = find_max_sum_column(Video_Match1)
        #Best_frame_match1 = max_Matchs1 / Select_frame
        Best_frame_offset1 = max_match_index1 - Frame_Windows
        OffsetOne[n+1] = Best_frame_offset1
        print(f"第[{Fnum_1}]号文件相比于第[{Fnum_0 }]号文件帧偏移为 {Best_frame_offset1}")
        OffsetAll[n+1] = OffsetAll[n] + Best_frame_offset1
        print(f"第[{Fnum_1}]号文件相比于第[{Firstname}]号文件帧偏移为 {OffsetAll[n+1]}")

    return OffsetOne,OffsetAll

