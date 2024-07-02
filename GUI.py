import tkinter as tk
from tkinter import messagebox

import time

from AllCal import CalSURF
from AllCal import CalSIFT
from AllCal import CalORB

#Select_File = 7#选择文件数
Select_frame = 150#匹配的有效帧数
Frame_Windows = 30#时域窗口大小
Input_frame = Select_frame + Frame_Windows * 2#匹配的读取帧数

First_point= 5000#提取特征点后根据响应度筛查后剩下的特征点
Final_point = 500#进行特征点筛选后剩下的有效特征点



def ORB(path):
    return [1, 2, 3]

def SIFT(path):
    return [4, 5, 6]


def calculate():
    path = entry_path.get()
    choiceCal = var1.get()
    choiceMode = var2.get()
    select_frame = int(entry_select_frame.get()) or Select_frame
    frame_windows = int(entry_frame_windows.get()) or Frame_Windows
    input_frame = select_frame + frame_windows * 2
    first_point =  First_point
    final_point =  Final_point
    select_file = int(entry_select_file.get()) - 1
    startcal = time.time()
    if choiceCal == 1:
        result = CalORB(path, select_frame, frame_windows, input_frame, first_point, final_point, choiceMode)
    elif choiceCal == 2:
        result = CalSIFT(path, select_frame, frame_windows, input_frame, first_point, final_point, choiceMode)
    elif choiceCal == 3:
        result = CalSURF(path, select_frame, frame_windows, input_frame, first_point, final_point, choiceMode)
    endcal = time.time()
    result  = [x - result[select_file] for x in result]
    messagebox.showinfo( '计算结果','相对于视频序列的第{}个视频，各视频帧偏移量为: {}，计算时间: {}'.format(select_file +1 ,result, endcal-startcal))

root = tk.Tk()
root.geometry('400x400')
root.title('多视角视频匹配')

tk.Label(root, text='请输入多视角视频文件路径:').pack()
entry_path = tk.Entry(root)
entry_path.pack()

var1 = tk.IntVar()
tk.Label(root, text='请选择你的匹配算法数字：').pack()
tk.Radiobutton(root, text='ORB(高效模式)', variable=var1, value=1).pack()
tk.Radiobutton(root, text='SIFT(精准模式)', variable=var1, value=2).pack()
tk.Radiobutton(root, text='SURF(平衡模式)', variable=var1, value=3).pack()

tk.Label(root, text='请输入视频同步的算法分析帧数:(推荐150）').pack()
entry_select_frame = tk.Entry(root)
entry_select_frame.pack()

tk.Label(root, text='请输入最大帧偏移量大小:(推荐30）').pack()
entry_frame_windows = tk.Entry(root)
entry_frame_windows.pack()

var2 = tk.IntVar()
tk.Label(root, text='请选择输入视频拍摄模式：').pack()
tk.Radiobutton(root, text='线性拍摄模式', variable=var2, value=1).pack()
tk.Radiobutton(root, text='环形拍摄模式', variable=var2, value=2).pack()

tk.Label(root, text='请输入参考视角编号:(视频名称顺序）').pack()
entry_select_file = tk.Entry(root)
entry_select_file.pack()



#tk.Label(root, text='请输入单帧初次筛查后的特征点数量:(推荐5000）').pack()
#entry_first_point = tk.Entry(root)
#entry_first_point.pack()

#tk.Label(root, text='请输入筛选后最终剩下的有效特征点数量：(推荐500）').pack()
#entry_final_point = tk.Entry(root)
#entry_final_point.pack()

tk.Button(root, text='计算', command=calculate).pack()

root.mainloop()

#pyinstaller --onefile GUI.py
#C:\MyIDE\Code\Python\fwktest2