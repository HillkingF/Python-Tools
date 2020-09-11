import os
import shutil
import cv2 as cv
import numpy as np

pritxt = 'C:\\Users\\office\\Desktop\\pritrain.txt'
newtxt = 'C:\\Users\\office\\Desktop\\newtrain.txt'
img = 'E:\\data\\90000shuibiao\\dz\\train_val\\'


with open(pritxt, 'r', encoding='UTF-8')as f:
    prilines = f.readlines()
    i = 0
    for priline in prilines:
        num = int(priline.split('.')[0].split('-')[-1])
        label = priline.split('.')[0].split('-')[-2]
        siglabel = priline.strip('\n').split(' ')[-1]

        fw = open(newtxt, 'a', encoding='utf-8')

        if label[num] == str(siglabel):
            fw.write(priline)
        else:
            i = i + 1
            print('\n')
            print('第 ' + str(i) + ' 张不一致的校验：' + siglabel)
            imgname = priline.split(' ')[0]
            src = cv.imdecode(np.fromfile(img + imgname, dtype=np.uint8), -1)
            cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
            cv.imshow('input_image', src)
            y = cv.waitKey(0)
            print(y)
            if y == 32:
                fw.write(priline)
            else:
                x = input('正确结果 ：')  # 正确的半字x
                fw.write(imgname + ' ' + x + '\n')
            cv.destroyAllWindows()

        fw.close()


