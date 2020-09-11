import os
import cv2
import numpy as np

def dic_ctc():
    letters_lower = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警领港澳学挂'
    label_dict = dict({i + 1: x for i, x in enumerate(list(letters_lower))})
    label_dict[0] = '-'


def label_dic(words):   # 制作label.txt时字符的对应关系
    letters_lower = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警领港澳学挂'
    # num_word = dict({i: x for i, x in enumerate(list(letters_lower))})  # 0:京
    word_num = dict({x: i for i, x in enumerate(list(letters_lower))})  # 京:0
    numbers = ''
    labels = ''
    for x in words:  #words = 皖ADX640
        numbers += str(word_num[x]) + '_'
        labels += str(word_num[x]) + ' '
    numbers = numbers[:-1] + '.jpg'   # numbers是转换后的图片名
    labels = labels[:-1]  # labels是转换后的标签
    return numbers, labels

def ccpd_dic(num):
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    plate = provinces[int(num[0])]  # 皖
    for x in num[1: ]:
        plate += ads[int(x)]
    return plate



if __name__ == '__main__':
    imgpath = 'E:\\data\\carplate\\newdata\\CCPD\\C\\ccpd_blur\\'
    saveimg = 'E:\\data\\carplate\\car_new_all\\CCPD\\ccpd_blur\\'
    savetxt = 'E:\\data\\carplate\\car_new_all\\CCPD\\ccpd_blur.txt'
    dir = 'ccpd_blur/'
    count = 331363
    for root, dirs, files in os.walk(imgpath):
        for file in files:
            count += 1
            # 1、新的图像名字 newimgname
            # 2、新的标签 labels
            number = file.split('-')[-3].split('_')
            ccpdplate = ccpd_dic(number)
            transname, labels = label_dic(ccpdplate)
            newimgname = str(count) + '-' + ccpdplate + '-' + transname
            # 3、新的坐标点points
            points = file.split('-')[3].replace('&', '_').split('_')
            # 4、裁剪但不保存
            img = cv2.imdecode(np.fromfile(imgpath+file, dtype=np.uint8), -1)
            pts1 = np.array(
                [[float(points[4]), float(points[5])],  # 左上
                 [float(points[6]), float(points[7])],  # 右上
                 [float(points[0]), float(points[1])],  # 右下
                 [float(points[2]), float(points[3])]],  # 左下
                dtype='float32')  # ,[float(self.zuobiao[3][0]),float(self.zuobiao[3][1])]
            pts2 = np.array([[0, 0], [256, 0], [256, 48], [0, 48]], dtype='float32')  # ,[0,48]
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img, M, (256, 48))
            # 5、保存图片并写txt
            cv2.imencode('.jpg', warped)[1].tofile(saveimg+newimgname)
            hang = dir + newimgname + ' ' + labels + '\n'
            print(hang)
            with open(savetxt, 'a', encoding='utf-8')as f:
                f.write(hang)