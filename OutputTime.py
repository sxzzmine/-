import time

def OutputT(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"本段代码执行时间: {elapsed_time} 秒")