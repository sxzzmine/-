from moviepy.editor import VideoFileClip
import ffmpeg
import os

def convert_flv_to_mp4(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".flv"):
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, os.path.splitext(file)[0] + ".mp4")
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

# 使用示例
input_path = 'C:\\Users\\weikangfu\\Desktop\\Work\\Shotcut\\Source'
output_path = 'C:\\Users\\weikangfu\\Desktop\\Work\\Shotcut\\Input'
convert_flv_to_mp4(input_path,output_path)