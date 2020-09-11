import os
import shutil

count = 0
img = 'E:\\data\\carplate\\car_new_all\\0_bg\\bg_collect\\'
newimg = 'E:\\data\\carplate\\car_new_all\\0_bg\\bg_train\\'
for root, dir, files in os.walk(img):
    for file in files:
        count  += 1
        line = 'bg_train/bg_collect_' + str(count) + '.jpg 0'
        print(line)
        # shutil.copy(img + file, newimg + 'bg_collect_' + str(count) + '.jpg')

exit()



txtmohu = 'E:\\data\\carplate\\car_new_all\\CCPD\\1-sort\\0bg\\ccpd.txt'
movemhimg = 'E:\\data\\carplate\\car_new_all\\CCPD\\1-sort\\0bg\\'    # 将所有模糊的进行合并

newdir = 'E:\\data\\carplate\\car_new_all\\0_bg\\'
count = 0
with open(txtmohu, 'r', encoding='utf-8')as f1:
    lines = f1.readlines()
    for line in lines:
        sortdir = line.split(' ')[0].split('/')[0]
        mohuname = line.split(' ')[0].split('/')[1]
        mohupath = movemhimg + sortdir + '\\' +mohuname

        if os.path.exists(mohupath):
            print(line)
            count += 1
            name = 'CCPD_' + str(count) + '.jpg'
            line = 'bg_train/' + name + ' 0\n'
            with open(newdir + 'new.txt', 'a', encoding='utf-8')as f:
                f.write(line)
                # shutil.copy(mohupath, newdir + 'bg_train\\' + name)
        else:
            pass
    print(count)


