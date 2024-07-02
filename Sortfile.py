
import os

def Sortexten(input_path,exten):
    nums_file = 0
    filelist = []
    for file in os.listdir(input_path):
        if file.endswith(exten):
            nums_file += 1
            filename, extension = os.path.splitext(file)
            filenum = int(filename)
            filelist.append((filenum, file))

    filelist.sort()
    sorted_filenames = [item[1] for item in filelist]

    print("Number of mp4 files:", nums_file)
    print("Sorted filenames:", sorted_filenames)
    return nums_file,sorted_filenames

