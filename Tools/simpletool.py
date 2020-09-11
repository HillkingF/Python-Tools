import os
import shutil
import cv2
import random
import numpy as np




# txt = 'E:/data/Test/waterplate/单字/142.txt'
# oldimg = 'E:/data/Test/waterplate/单字/裁剪后图片142张/'
# newimg = 'E:/data/Test/waterplate/单字/cut/'
# with open(txt, 'r', encoding='utf-8')as f:
#     lines = f.readlines()
#     sign = 0
#     imgname = ''
#     newimgname = ''
#     for line in lines:
#         if line.endswith('.jpg\n'):
#             count = 0
#             imgname = line.strip('\n')
#         elif len(line.strip('\n')) != 1:
#             print(line)
#             count += 1
#             newimgname = imgname.strip('.jpg') + '_' + str(count) + '.jpg'
#             loc = line.split(' ')[0: -1]
#             # 下面开始裁剪图像
#             img = cv2.imdecode(np.fromfile(oldimg + imgname, dtype=np.uint8), -1)
#             pts1 = np.array(
#                 [[float(loc[0]), float(loc[1])],  # 左上
#                  [float(loc[0])+ float(loc[2]), float(loc[1])],  # 右上
#                  [float(loc[0]) + float(loc[2]), float(loc[1])+ float(loc[3])],  # 右下
#                  [float(loc[0]), float(loc[1])+float(loc[3])]],  # 左下
#                 dtype='float32')
#             pts2 = np.array([[0, 0], [128, 0], [128, 32], [0, 32]], dtype='float32')  # ,[0,48]
#             M = cv2.getPerspectiveTransform(pts1, pts2)
#             warped = cv2.warpPerspective(img, M, (128, 32))
#             cv2.imencode('.jpg', warped)[1].tofile(newimg + newimgname)




