import pandas as pd
import os
import shutil
from glob import glob
def mycopyfile(srcfile, dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))

dataset = 'O_12195'
number = '10001-12195'
data = pd.read_csv('result/' + str(dataset) + '/' + str(number) + '/RT_right_results.csv').values
# dst_dir1 = 'copy-data/' + str(dataset) + '/cif/'
# dst_dir2 = 'copy-data/' + str(dataset) + '/txt/'
# dst_dir3 = 'copy-data/' + str(dataset) + '/txt/'
dst_dir4 = 'origin_net/' + str(dataset) + '/'
print(data.shape)
data_lists = []
for i in range(len(data)):
    data_lists.append(data[i][0])

for srcfile in data_lists:
    # srcfile1 = 'data/' + str(dataset) + '/' + srcfile + '.cif'
    # srcfile2 = 'data/' + str(dataset) + '/' + srcfile + '_adjacency_channel_atoms2.txt'
    # srcfile3 = 'data/' + str(dataset) + '/' + srcfile + '_adjacency_void_atoms2.txt'
    srcfile4 = 'mid_out/' + str(dataset) + '/' + str(number) + '/' + srcfile + '_origin.net'

    # mycopyfile(srcfile1, dst_dir1)
    # mycopyfile(srcfile2, dst_dir2)
    # mycopyfile(srcfile3, dst_dir3)
    mycopyfile(srcfile4, dst_dir4)
