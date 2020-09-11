### 图像增强的两个库 imgaug 和 augmentotr
### imgaug 【https://github.com/aleju/imgaug】  【http://imgaug.readthedocs.io】  里面有全部的安装教程、卸载教程

# from imgaug import augmenters as iaa
# seq = iaa.Sequential([  # 首先定义一个变幻序列
#     # iaa.Fliplr(0.5),
#     #     # iaa.GaussianBlur(sigma=(0,30))
#     iaa.PiecewiseAffine(scale=(0.001,0.005))
# ])
#
# image_aug = seq.augment_image('1.jpg')
from imgaug import augmenters as iaa
import cv2
import numpy as np
import imgaug as ia

im = cv2.imread('E:/Product/untitled/1.jpg')
im = cv2.resize(im,(256, 256)).astype(np.int8)
images = np.zeros([16,256,256,3])
images[0] = im

# st = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.3),
    iaa.GaussianBlur((1,2.0))
    # st(iaa.PiecewiseAffine(scale=(0.003,0.005)))
])

images_aug = seq.augment_images(images[0:100,0:100])
seq.show_grid(images[0],rows = 1,cols=1)



