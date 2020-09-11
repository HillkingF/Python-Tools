import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import os

df = pd.read_csv('dz_bz.txt', delimiter='\t', header=None, index_col=None)
df.index = df.index + 1
df['filename'], df['ban'], df['quan'], df['mohu'] = df[0].str.split(' ', 3).str
del df[0]

dir = 'cuowu'
count = 0  # 记录处理文件数量
processed_files = []  # 记录已处理的图片文件
error_files = []  # 错误的文件

for file in sorted(os.listdir(dir), key=lambda x: int(x.split('_')[0])):
    # try:
    print('---------------------------')
    filepath = 'cuowu/' + file
    print(filepath)
    # 显示图片
    img = io.imshow(filepath)
    plt.show()
    # 图片名前缀
    print(file)
    fname = file.split('_')[1].split('-')[0] + '-' + file.split('_')[1].split('-')[1]
    print(df[df['filename'] == file.split('_')[1]])
    # 获取输入
    get_input = input()
    if get_input == 'q':  # 跳过
        tmp_df = df.loc[df['filename'] == file.split('_')[1]]
    elif get_input == 's':  # 模糊
        df.mohu[df['filename'].str.startswith(fname)] = 's'
    elif get_input == 'p':  # 退出
        break
    else:  # 数字
        df.ban[df['filename'] == file.split('_')[1]] = get_input
        print(get_input)
    print(df[df['filename'].str.startswith(fname)])
    # 保存
    df.to_csv('dz_bz.txt', index=False, header=False, sep=' ')
    plt.close(1)

    # 统计信息
    count += 1
    processed_files.append(filepath)

    # except Exception as err:
    #     print('出错：' + str(err))
    #     error_files.append(filepath)

with open('processing_log.txt', 'a') as f:
    f.write('----------------------------------统计如下：--------------------------------------\n')
    f.write('已处理文件总数：')
    f.write(str(count))
    f.write('\n已处理文件名：')
    for ff in processed_files:
        f.write(str(ff) + '\n')
    f.write('\n\n\n')
    f.write('\n出错的文件名：')
    for ff in error_files:
        f.write(str(ff) + '\n')
