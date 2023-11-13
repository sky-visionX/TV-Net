import os

import shutil

if __name__ == "__main__":
    b_file = "/media/rmz/749d3547-4bb1-4302-a40c-63292a590f0a/data/2D公开道路裂缝数据集/DeepCrack-datasets/CRKWH100/CRKWH100_gt/"
    b_name = "CRKWH100_"
    e_file = "/media/rmz/749d3547-4bb1-4302-a40c-63292a590f0a/data/2D公开数据集整理/train/masks/"
    files = os.listdir(b_file)
    for file in files:
        file_name = os.path.basename(file)
        file_name1 = b_name+file_name
        #src = 'A.py'
        #dst = 'Folder/B.py'
        #print("file",file_name)

        src = os.path.join(b_file,file_name)
        dst = os.path.join(e_file,file_name1)
        shutil.copyfile(src=src, dst=dst)
        # 将当前路径下的 A.py 拷贝到Folder下另存为 B.py, 不复制元数据。
        # 但是，如果它已经存在，它会覆盖目标文件。

