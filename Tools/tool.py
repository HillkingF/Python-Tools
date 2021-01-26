# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import shutil
import random
from random import randint
from PIL import Image
from xml.dom.minidom import parse
import xml.dom.minidom
import logging
import sys
import pandas as pd




class toolclass:

    def __init__(self):
        super().__init__()
        self.txtfile = None
        self.a = []
        self.relatelines = []
        # self.userwindow()

    # 输入某一字符串，查找指定目录下是否包含此字符串,返回所有包含此字符串的行的列表集合
    def findlines(self, priline, objpath):
        self.relatelines = []
        if os.path.isfile(objpath):
            file = open(objpath, 'r', encoding='UTF-8')
            lines = file.readlines()
            for line in lines:
                if priline in line:
                    self.relatelines.append(line)
            file.close()
            return self.relatelines
        else:
            return ''
    def writeline(self, strs, file):
        file.write(strs)

    # 越模糊的好像值越低； 越亮的图像得到的结果越高，越暗的图像结果越低。所以极大值和极小值可能都是模糊的
    def getImageVar(self,imgPath):
        image = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
        # image = cv2.imread(imgPath)
        img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()   # 计算方差并返回
        return imageVar

    # 计算低亮度像素在整个图像中的范围
    def is_low_contrast(self,img, fraction_threshold=0.5, lower_percentile=1, upper_percentile=99):
        src = Image.open(img)
        # 转换为灰度图像
        src = np.asanyarray(src)
        if src.ndim == 3 and src.shape[2] in [3, 4]:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # 获取图像数据类型的最大值和最小值
        dlimits = (np.iinfo(src.dtype).min, np.iinfo(src.dtype).max)
        # 计算低亮度像素在整个图像中的范围
        limits = np.percentile(src, [lower_percentile, upper_percentile])
        # 计算比例
        ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
        return ratio
    def contrast(self, img0):
        img1 = cv2.imdecode(np.fromfile(img0, dtype=np.uint8), -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
        m, n = img1.shape
        # 图片矩阵向外扩展一个像素
        img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        rows_ext, cols_ext = img1_ext.shape
        b = 0.0
        for i in range(1, rows_ext - 1):
            for j in range(1, cols_ext - 1):
                b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 + (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)
        cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
        return cg
    def walkDir(self,rootpath,tarpath):
        for root, dirs, files in os.walk(rootpath):
            for file in files:
                print(os.path.join(root, file))  # 写出这个文件的完整路径
                img0 = cv2.imread(rootpath + file)
                data = self.contrast(img0)
                if data < 26000 and data > 25000:
                    shutil.move(rootpath + file, tarpath)
                    print(file, data)
    def filter(self, imgpath):#按照分辨率挑选
        img = Image.open(imgpath)
        imgSize = img.size
        maxSize = max(imgSize)
        minSize = min(imgSize)
        img.close()
        return maxSize
        # newimgpath = moveimgpath
        # if maxSize <100 or minSize < 30:
        #     shutil.move(imgpath, newimgpath)
    def jcsort(pritxt, newtxt):  # 对检测的数据排序
        with open(pritxt, 'r', encoding='utf-8')as f, open(newtxt, 'w', encoding='utf-8')as f1:
            lines = f.readlines()
            imgname = ''
            num = 0
            coordinate = dict()
            for line in lines:
                cline = line.strip('\n')
                if cline.endswith('.jpg'):
                    imgname = line
                elif len(cline) == 1:
                    num = int(cline)
                else:
                    zbvalue = cline.split(' ')[0:4]  # 含左不含右
                    zbkey = float(zbvalue[0])  # float
                    coordinate.update({zbkey: zbvalue})  # 将原始坐标行存入字典

                if len(coordinate) == num & num != 0:

                    # 排序
                    keylst = []
                    for k in coordinate.keys():
                        keylst.append(k)
                    keylst.sort()
                    newcoor = dict()
                    for x in keylst:
                        newcoor.update({x: coordinate[x]})

                    # 写txt
                    f1.write(imgname)  # 写入图片名
                    f1.write(str(num) + '\n')  # 写入个数
                    for v in newcoor.values():  # 写入坐标
                        zb = str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + ' ' + str(v[3]) + '\n'
                        f1.write(zb)

                    # 初始化
                    coordinate = {}

    # 随机挑选不同数量的标签，比如txt中随机选100行
    def select_someof_label(self, pritxt, newtxt, labelnum):
        with open(pritxt, 'r', encoding='utf-8')as f1:
            f1lines = f1.readlines()
        with open(newtxt, 'w', encoding='utf-8')as f2:
            for _ in range(labelnum):
                f2.write(f1lines.pop(random.randint(0,len(f1lines)-1)))

    # 这个函数主要是借鉴字典的使用
    def dicues(self):
        # 统计标签中每一类的标签个数
        value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        label_dict = dict({str(x): i for i, x in enumerate(value)})
        num = dict({i: 0 for i in range(20)})
        label = 'C:\\Users\\office\\Desktop\\0label.txt'
        with open(label, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                dzlabel = line.strip('\n').split(' ')[1]
                num[label_dict[dzlabel]] += 1
            sum = 0
            for j in range(20):
                sum = sum + num[j]
            print(sum)

    def shaixuan(self, txt1, txt2, txt3):
        self.sign = 0
        f3 = open(txt3, 'w', encoding='utf-8')
        with open(txt1, 'r', encoding='utf-8')as f1, open(txt2, 'r', encoding='utf-8')as f2:
            f1lines = f1.readlines()
            f2lines = f2.readlines()

        for f1line in f1lines:
            if f1line.endswith('.jpg\n'):
                self.sign = 0
                # 进入另外一个文件搜索不同
                for f2line in f2lines:
                    name = f2line.split(' ')[0]
                    if f1line.strip('\n') in name:
                        self.sign = 1
                        break
            if self.sign == 0:
                print(f1line)
                f3.write(f1line)

        f3.close()

    def matchAndsort(self, pritxt, pritest, newtxt):
        sign = False
        num = 0
        temp = 0
        with open(pritxt, 'r', encoding='utf-8')as f1:
            f1lines = f1.readlines()
        with open(pritest, 'r', encoding='utf-8')as f2:
            f2lines = f2.readlines()
        f3 = open(newtxt, 'w', encoding='utf-8')
        for f1line in f1lines:    # 读取的是检测的test
            if f1line.endswith('.jpg\n'):
                sign = False
                str = f1line.strip('\n')
                for f2line in f2lines:   # 遍历含有标签的test
                    if str in f2line:
                        num += 1
                        sign = True
                        label = f2line.strip('\n').split(' ')[-1]
                        f1line = f1line.replace('\n', ' ') + label + '\n'
                        temp = 1
                if temp == 0:
                    print(f1line.strip('\n'))
                temp = 0
            if sign == True:
                f3.write(f1line)
        print(num)
        f3.close()

    def sorttxt(self, pritxt, newtxt):
        hangshu = 0    # 1
        xsort = []     # 2
        zuobiao = dict()   # 3

        with open(pritxt, 'r', encoding='utf-8')as f1:
            f1lines = f1.readlines()
        f2 = open(newtxt, 'w', encoding='utf-8')
        for f1line in f1lines:
            if f1line.split(' ')[0].endswith('.jpg'):    # 这个判断用于获取标签
                hangshu = 0   # 1
                xsort = []  # 2
                zuobiao = dict()  # 3
                label = f1line.strip('\n').split(' ')[-1]
                length = len(label)
                f2.write(f1line)
            elif len(f1line.split(' ')) == 5:
                hangshu += 1
                xsort.append(float(f1line.split(' ')[0]))
                zuobiao.update({float(f1line.split(' ')[0]):f1line})
            if hangshu == length:
                xsort = sorted(xsort)
                for x in xsort:
                    f2.write(zuobiao[x])
        f2.close()  # 写文件的关闭

    # 裁剪图片---保存图片self.parameters()
    def cropimg(self, priimgpath, newimgpath):
        img = Image.open(priimgpath)  #'E:\\data\\90000shuibiao\\allimg\\10000-00149.jpg'
        cropped = img.crop((76.854, 2.656, 76.854 + 38.053, 2.656 + 32.384))  # (left, upper, right, lower)
        # cropped = img.crop((x, y, x + w, y + h))  # (left:x, upper:y, right:x+w, lower:y+h)
        cropped.save(newimgpath)

    # 随机  按比例  抽取图片
    def moveFile(fileDir):
        pathDir = os.listdir(fileDir)  # 取图片的原始路径
        filenumber = len(pathDir)
        rate = 0.00611245  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
        print(sample)
        print(len(sample))
        for name in sample:
            shutil.move(fileDir + name, tarDir + name)
        return

    # 处理xml文件   使用xml.dom解析xml  解析一个 XML 文档时，一次性读取整个文档，把文档中所有元素保存在内存中的一个树结构里
    def xmlanlyse(self, xmlpath):

        # 使用minidom解析器打开xml文档

        DOMTree = parse(xmlpath)
        collection = DOMTree.documentElement
        moakeys = collection.getElementsByTagName('MOAKey')
        self.numbers = []  # 车牌号 ObjectText
        self.danshuangs = [] # 单双排字符串
        self.coordinates = []  # 坐标
        self.ids = []  # 编号，用于区分图像
        self.rtn = []    # 最后返回的列表
        self.types = []  # 暂时没用到
        for moakey in moakeys:  # 获取每一段moakeys的内容
            zuobiao = []
            try: # objectText有车牌号
                number = moakey.getElementsByTagName('ObjectText')[0].childNodes[0].data.replace(' ', '')
                type = '2'
            except Exception as err:
                type = '2'
                number = '无'
            lable = moakey.getElementsByTagName('Lable')[0]
            try:
                text = lable.getElementsByTagName('Text')[0].childNodes[0].data
            except Exception as err:
                text = '无色层'
            id = moakey.getElementsByTagName('Id')[0].childNodes[0].data
            points = moakey.getElementsByTagName('Points')[0] # moakey获取的是一个数组，0表示数组中的第0个元素
            pointn = points.getElementsByTagName('Point')   # 获取Points中所有的Point
            for point in pointn:
                x = point.getElementsByTagName('X')[0].childNodes[0].data
                y = point.getElementsByTagName('Y')[0].childNodes[0].data
                zuobiao.append([x,y])
            if len(zuobiao)== 5:
                self.coordinates.append(zuobiao)  # zuobiao  一组一定有五个
                self.numbers.append(number)  # 车牌号    无
                self.ids.append(id)
                self.types.append(type)
                self.danshuangs.append(text)  # 车牌颜色类型

        self.rtn.append(self.coordinates)
        self.rtn.append(self.ids)
        self.rtn.append(self.types)
        self.rtn.append(self.numbers)
        self.rtn.append(self.danshuangs)
        return self.rtn

    # 根据四个角点裁剪图片  这个方法用的是透射变换
    def cutpic_4points(self, oldpath, zuobiao, sizechange):
        try:
            if os.path.exists(oldpath[0]):#oldpath[0]):
                img = cv2.imdecode(np.fromfile(oldpath[0], dtype=np.uint8), -1)
            elif os.path.exists(oldpath[1]):
                img = cv2.imdecode(np.fromfile(oldpath[1], dtype=np.uint8), -1)
            # 2、可以在这里进行图像扩增
            if sizechange == False:
                zuox = 0
                youx = 0
                shangy = 0
                xiay = 0
            else:  # sizechange == True:
                zuox = random.randint(-2, 17)
                youx = random.randint(-2, 17)
                shangy = random.randint(-2, 17)
                xiay = random.randint(-2, 17)
            # 3、计算图片的高度和宽度
            pts1 = np.array(
                [[float(zuobiao[0][0].split('.')[0]) - zuox, float(zuobiao[0][1].split('.')[0]) - shangy],  # 左上
                 [float(zuobiao[1][0].split('.')[0]) + youx, float(zuobiao[1][1].split('.')[0]) - shangy],  # 右上
                 [float(zuobiao[2][0].split('.')[0]) + youx, float(zuobiao[2][1].split('.')[0]) + xiay],  # 右下
                 [float(zuobiao[3][0].split('.')[0]) - zuox, float(zuobiao[3][1].split('.')[0]) + xiay]],  # 左下
                dtype='float32')  # ,[float(self.zuobiao[3][0]),float(self.zuobiao[3][1])]
            pts2 = np.array([[0, 0], [256, 0], [256, 48], [0, 48]], dtype='float32')  # ,[0,48]
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img, M, (256, 48))
            return warped
            # cv2.imencode('.jpg', warped)[1].tofile(newpath)
        except Exception as err:
            return 'cut_error'

    # 根据左上和右下两个点裁剪图像，使用的还是opencv
    def cutpic_2points(self, picpath, savepath, zuoshang_x, zuoshang_y, youxia_x, youxia_y):
        img = cv2.imdecode(np.fromfile(picpath, dtype=np.uint8), -1)
        coor1x = int(zuoshang_x)  # 20  向下取整
        coor1y = int(zuoshang_y)  # 30
        coor3x = int(youxia_x) + 1  # 80  向上取整
        coor3y = int(youxia_y) + 1  # 90
        crop = img[coor1y: coor3y, coor1x: coor3x, :]  # 30:90, 20:80,  第三个：应该是指通道
        cv2.imwrite(savepath, crop)



    # 随机从txt挑选一定数量的行，进行数据集划分
    def suiji_select(self, pritxt, selectpath, restpath, selectcount):
        # 读取原来全部的label
        f1 = open(pritxt, 'r', encoding='utf-8')
        allrows = f1.readlines()
        # 从所有行中选择
        self.resultList = sorted(random.sample(range(0, len(allrows)),selectcount))    # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。
        selecttxt = open(selectpath, 'w', encoding='utf-8')
        resttxt = open(restpath, 'w', encoding='utf-8')
        for i, line in enumerate(allrows):
            print(i)
            if i in self.resultList:
                selecttxt.write(line)
            else:
                resttxt.write(line)
        f1.close()
        selecttxt.close()
        resttxt.close()

    # 日志模块
    def logs(self, savepath): #, time, prov, type,
        logger = logging.getLogger('test_')  # 定义对应的程序模块名name，默认是root
        file = savepath + '.log'

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=file,
                            filemode='w')

        # 1、设置输出目的地和输出最低日志级别
        console = logging.StreamHandler()  # 日志输出到屏幕控制台
        console.setLevel(logging.DEBUG) #设置日志等级
        fh = logging.FileHandler(file)  # 向文件access.log输出日志信息
        fh.setLevel(logging.DEBUG)  # 设置输出到文件最低日志级别
        # 2、定义日志输出格式
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # 3、add formatter to console and fh
        console.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(console)
        logger.addHandler(fh)
        # 4、测试
        logger.debug('New debug message')
        logger.info('New info message')
        logger.warning('New warn message')
        logger.error('New error message')
        logger.critical('New critical message')
        return logger

    # pandas
    def pandas_using(self): # 相关与excel,使用 dataframe完成需要的功能
        # 读取数据
        data = pd.read_csv(test.csv)
        data.to_csv('test.csv',index=None)
    def createdir(self, dirpath):   # 创建目录
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def drawimg_withpoints(self):  # 在图像上画点
        from PIL import Image
        from pylab import imshow
        from pylab import array
        from pylab import plot
        from pylab import title
        import pylab
        # 读取图像到数组中
        im = array(Image.open(r'D:\GitHub\Python-Tools\Tools\ballet_106_0_0.jpg'))
        # 绘制图像
        imshow(im)
        # 一些点
        x = 13.786744992  #[box[0][0], box[1][0], box[2][0], box[3][0], ]
        y = 6.24719206297                #[box[0][1], box[1][1], box[2][1], box[3][1], ]
        # 使用红色星状标记绘制点
        plot(x, y, 'r*')

        # 绘制连接前两个点的线
        # plot(x[:2],y[:2])
        # 添加标题，显示绘制的图像
        title('Plotting: "empire.jpg"')
        pylab.show()

    def grayimg(self,image): # 将一幅图像变成灰度图，并放缩到特定的长度
        img = cv2.imread(image, 0)
        # img = cv2.imread(image)
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        cv2.imshow("image", img)
        dst = cv2.resize(img, (50, 8), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("imd",dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__ == '__main__':

    # txt36 = r'E:\data\carplate\fangdahao\重复车尾数据.txt'
    # newcut = r'E:\data\carplate\fangdahao\第2批-20201221车尾放大号标注数据-共960个标签\20201221车尾放大号标注数据\newcut.txt'
    # txt1 = r'E:\data\carplate\fangdahao\第2批-20201221车尾放大号标注数据-共960个标签\20201221车尾放大号标注数据\cut.txt'
    # with open(txt36,'r',encoding='utf-8')as f1, open(txt1,'r',encoding='utf-8')as f2,\
    #     open(newcut, 'w', encoding='utf-8')as f3:
    #     f1lines = f1.readlines()
    #     f2lines = f2.readlines()
    #     count = 0
    #
    #     for f2line in f2lines:
    #         sign = 0
    #         count += 1
    #         print(count)
    #         name = f2line.split(' ')[0][0:-6]
    #         print(name)
    #         for f1line in f1lines:
    #             if name in f1line:
    #                 sign = 1
    #                 break
    #         if sign == 0:
    #             f3.write(f2line)
    #         sign = 0
    #
    # exit()


    # # 随机选出300张当作验证集
    # txt = r'E:\data\carplate\fangdahao\dataset\all.txt'
    # train = r'E:\data\carplate\fangdahao\dataset\txttrain.txt'
    # val = r'E:\data\carplate\fangdahao\dataset\txtval.txt'
    #
    # oldimg = r'E:\data\carplate\fangdahao\dataset\all' + '\\'
    # valimg = r'E:\data\carplate\fangdahao\dataset\val' + '\\'
    # trainimg = r'E:\data\carplate\fangdahao\dataset\train' + '\\'
    #
    # resultList = sorted(random.sample(range(0, 4348), 300))
    # with open(txt, 'r', encoding='utf-8')as f1, open(train, 'w',encoding='utf-8')as f2, open(val, 'w',encoding='utf-8')as f3:
    #     f1lines = f1.readlines()
    #     count = 0
    #     for i, line in enumerate(f1lines):
    #         name = line.split(' ')[0]
    #         if i in resultList:
    #             count += 1
    #             print(count)
    #             try:
    #                 shutil.move(oldimg + name, valimg + name)
    #                 f3.write(line)
    #
    #             except:
    #                 print(line)
    #         else:
    #             try:
    #                 shutil.move(oldimg + name, trainimg + name)
    #                 f2.write(line)
    #
    #             except:
    #                 print(line)
    # exit()





    shengstr = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警领港澳学挂'
    label_dict = dict({x: i for i, x in enumerate(list(shengstr))})

    txt = r'E:\data\carplate\fangdahao\dataset\test.txt'
    newtxt = r'E:\data\carplate\fangdahao\dataset\newtest.txt'

    with open(txt, 'r', encoding='utf-8')as f1, open(newtxt, 'a', encoding='utf-8')as f2:
        lines = f1.readlines()
        for line in lines:
            name = line.split(' ')[0]
            num = line.strip('\n').split(' ')[1]
            hang = name
            try:
                for x in num:
                    hang = hang + ' ' + str(label_dict[x])
            except:
                print(line)
                continue
            hang = 'test/' + hang + '\n'
            f2.write(hang)


    exit()

    txtdir = r'E:\data\carplate\fangdahao\第1批-测试数据-150张图片\第一批-测试数据-150张图片\原始压缩包\txt' + '\\'
    imgdir = r'E:\data\carplate\fangdahao\第1批-测试数据-150张图片\第一批-测试数据-150张图片\原始压缩包\jpg' + '\\'
    cutimg = r'E:\data\carplate\fangdahao\第1批-测试数据-150张图片\第一批-测试数据-150张图片\原始压缩包\cut' + '\\'
    newtxt = r'E:\data\carplate\fangdahao\第1批-测试数据-150张图片\第一批-测试数据-150张图片\原始压缩包\cut.txt'

    for root, dir, files in os.walk(txtdir):
        for file in files:
            with open(txtdir + file, 'r', encoding='utf-8')as f1:
                lines = f1.readlines()
                linecount = 0
                for line in lines:
                    linecount += 1
                    zuox = float(line.split(' ')[1])
                    zuoy = float(line.split(' ')[2])
                    carplate = line.strip('\n').split(' ')[-1]
                    w = float(line.split(' ')[3])
                    h = float(line.split(' ')[4])

                    oldimg = imgdir + file.split('.')[0] + '.jpg'
                    newimg = cutimg + file.split('.')[0] + '-' + str(linecount) + '.jpg'

                    hang = file.split('.')[0] + '-' + str(linecount) + '.jpg' + ' ' + carplate + '\n'

                    img = cv2.imdecode(np.fromfile(oldimg, dtype=np.uint8), -1)

                    sizechange = True
                    if sizechange == False:
                        jzuox = 0
                        jyoux = 0
                        jshangy = 0
                        jxiay = 0
                    else:  # sizechange == True:
                        jzuox = random.randint(-2, 24)
                        jyoux = random.randint(-2, 24)
                        jshangy = random.randint(-2, 20)
                        jxiay = random.randint(-2, 20)
                    pts1 = np.array(
                        [[zuox - jzuox , zuoy - jshangy],  # 左上
                         [zuox + w + jyoux, zuoy - jshangy],  # 右上
                         [zuox + w + jyoux, zuoy + h + jxiay],  # 右下
                         [zuox - jzuox, zuoy + h + jxiay]],  # 左下
                        dtype='float32')
                    pts2 = np.array([[0, 0], [256, 0], [256, 48], [0, 48]], dtype='float32')  # ,[0,48]
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    warped = cv2.warpPerspective(img, M, (256, 48))
                    cv2.imencode('.jpg', warped)[1].tofile(newimg)
                    with open(newtxt, 'a', encoding='utf-8')as f2:
                        f2.write(hang)
    exit()





















    exit()
    noccpdtxt = 'E:\\data\\carplate\\car_all\\2_yuanshi\\clear_yuanshi.txt'
    outtxt = 'E:\\data\\carplate\\car_all\\2_yuanshi\\out.txt'
    resttxt = 'E:\\data\\carplate\\car_all\\2_yuanshi\\rest.txt'
    imgdir = 'E:/data/carplate/car_all/2_yuanshi/'
    wcount = 0
    hcount = 0
    sign = 0
    with open(noccpdtxt, 'r', encoding='utf-8')as f1, \
        open(outtxt, 'w', encoding='utf-8')as f2, \
        open(resttxt, 'w', encoding='utf-8')as f3:
        lines = f1.readlines()
        # 初始化
        name = lines[0].split(' ')[0]
        img = Image.open(imgdir + name)
        imgSize = img.size
        maxs= max(imgSize)
        mins = min(imgSize)
        img.close()
        for line in lines:
            sign = 0
            name = line.split(' ')[0]
            # 按照大小过滤
            img = Image.open(imgdir+name)
            imgSize = img.size
            maxSize = max(imgSize)
            minSize = min(imgSize)
            img.close()
            # 记录最大最小像素并选出小分辨率图像：sign=1
            if maxSize > maxs:
                maxs = maxSize
            if minSize < mins:
                mins = minSize
            if maxSize < 100 or minSize < 25:
                sign = 1
            # 计算模糊数值
            mohu = usetool.getImageVar(imgdir+name)
            if mohu < 40 or mohu > 1500:
                sign = 1
            # 计算低亮度像素在整个图像中的范围
            ratio = usetool.is_low_contrast(imgdir+name)
            if ratio < 0.5 or ratio > 0.9:
                sign = 1
            print(sign,  maxSize, minSize, mohu, ratio)
            if sign == 1:
                f2.write(line)
            else:
                f3.write(line)

        print(maxs, mins)
    exit()














    '''处理xml'''
    ### 原始xml和img
    xmlpath = 'E:/data/carplate/car_dirty_cover/car_dirty_cover/car_wusun/第二批/500jiance/OAT/'
    oldpath = 'E:/data/carplate/car_dirty_cover/car_dirty_cover/car_wusun/第二批/500jiance/'

    # 一些特殊字符的车牌 直接不放缩切出来
    specialimg = 'E:/data/carplate/car_amplification/'+times+'/special/special/'
    specialtxt = 'E:/data/carplate/car_amplification/'+times+'/special/speciallabel.txt'
    specialfile = open(specialtxt, 'w', encoding='utf-8')

    # 字典
    letters_lower = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警领港澳学挂'
    newdic = dict({x: i for i, x in enumerate(list(letters_lower))})
    count = -1
    num = 0
    for root, dirs, filenames in os.walk(xmlpath):
        for filename in filenames:
            if filename.endswith('.oa'):  # 判断是否是oa文件
                # 解析xml： ①原始图片  新图片  路径； ②到xml中获取坐标信息 ③④⑤返回值分别是坐标组、id组、type组
                xml = xmlpath + filename
                rtn = usetool.xmlanlyse(xml)  # 返回值分别是：rtn[坐标组，id组，type组，车牌号组]
                zuobiaos = rtn[0]
                ids = rtn[1]
                type = rtn[2]
                number = rtn[3]
                danshuang = rtn[4]
                if len(zuobiaos) != 0:
                    for i,x in enumerate(zuobiaos):
                        # print(num, filename)
                        num += 1

                if len(zuobiaos) == 0:
                    continue
                else:
                    for i, coordinates in enumerate(zuobiaos):
                        try:  # 对一组点开始进行变换_处理一个车牌
                            oldjpg = oldpath + filename.replace('.oa', '.jpg')  # 加.是为了避免图片名中也含有字符oa
                            oldjpeg = oldpath + filename.replace('.oa', '.jpeg')
                            oldimg = [oldjpg, oldjpeg]
                            if os.path.exists(oldimg[0]) or os.path.exists(oldimg[1]):
                                count += 1
                                print(count)
                                # 若车牌号中有不是字典中的字符，则添加到special目录下
                                carsign = True
                                for x in number[i]:
                                    if (x in newdic.keys()) and (len(number) != 0):
                                        pass
                                    else:
                                        carsign = False

                                # 根据信号选择切图保存之处
                                if carsign == True: # 车牌号都是字典中的
                                    # 原始尺寸的剪切填写
                                    normalcutres = usetool.cutpic(oldimg, coordinates, sizechange=False)
                                    if normalcutres is not 'cut_error':
                                        normalhang = str(count) + '_' + number[i] + '.jpg'  # 图片名
                                        numbertype = []
                                        for x in number[i]:
                                            numbertype.append(newdic[x])
                                        for x in numbertype:
                                            normalhang = normalhang + ' ' + str(x)
                                        normalhang = normalhang + ' ' + type[i] + '\n'
                                        print(normalhang)
                                        if '双排' in danshuang[i]:
                                            normalimg = 'E:/data/carplate/car_amplification/' + times + '/double/normalimg/'
                                            cv2.imencode('.jpg', normalcutres)[1].tofile(normalimg + str(count) + '_' + number[i] + '.jpg')
                                            normaltxt = 'E:/data/carplate/car_amplification/' + times + '/double/normallabel.txt'
                                            with open(normaltxt, 'a', encoding='utf-8')as f1:
                                                pass
                                                f1.write(normalhang)
                                        else:
                                            normalimg = 'E:/data/carplate/car_amplification/' + times + '/single/normalimg/'
                                            cv2.imencode('.jpg', normalcutres)[1].tofile(normalimg + str(count) + '_' + number[i] + '.jpg')
                                            normaltxt = 'E:/data/carplate/car_amplification/' + times + '/single/normallabel.txt'
                                            with open(normaltxt, 'a', encoding='utf-8')as f2:
                                                f2.write(normalhang)
                                                pass
                                    # 缩放尺寸的剪切填写
                                    # suofangimgname = suofangimg + str(count) + '_' + number[i] + '.jpg'
                                    suofangcutres = usetool.cutpic(oldimg, coordinates, sizechange=True)
                                    if suofangcutres is not 'cut_error':
                                        suofanghang = str(count) + '_' + number[i] + '.jpg'  # 图片名
                                        numbertype = []
                                        for x in number[i]:
                                            numbertype.append(newdic[x])
                                        for x in numbertype:
                                            suofanghang = suofanghang + ' ' + str(x)
                                        suofanghang = suofanghang + ' ' + type[i] + '\n'
                                        if '双排' in danshuang[i]:
                                            suofangimg = 'E:/data/carplate/car_amplification/' + times + '/double/suofang/'
                                            cv2.imencode('.jpg', suofangcutres)[1].tofile(suofangimg+str(count) + '_' + number[i] + '.jpg')
                                            suofangtxt = 'E:/data/carplate/car_amplification/' + times + '/double/suofanglabel.txt'
                                            with open(suofangtxt, 'a', encoding='utf-8')as f3:
                                                f3.write(suofanghang)
                                        else:
                                            suofangimg = 'E:/data/carplate/car_amplification/' + times + '/single/suofang/'
                                            cv2.imencode('.jpg', suofangcutres)[1].tofile(suofangimg+str(count) + '_' + number[i] + '.jpg')
                                            suofangtxt = 'E:/data/carplate/car_amplification/' + times + '/single/suofanglabel.txt'
                                            with open(suofangtxt, 'a', encoding='utf-8')as f4:
                                                f4.write(suofanghang)
                                else:  # 车牌号不在字典中
                                    if len(number[i]) != 0:  # 有车牌号
                                        specialimgname = specialimg + str(count) + '_' + filename.replace('.oa', '.jpg')
                                        specialcutres = usetool.cutpic(oldimg, coordinates, sizechange=False)
                                        if specialcutres is not 'cut_error':
                                            sphang = str(count) + '_' + filename.replace('.oa', '.jpg ') + str(number[i])+'\n'# 图片名
                                            specialfile.write(sphang)
                                            cv2.imencode('.jpg', specialcutres)[1].tofile(specialimgname)

                            else:
                                with open('nopicture.txt', 'a') as f:
                                    f.write(filename + '\n')

                        except Exception as err:
                            print('发生错误！ ' + oldimg + ':' + str(err))
                            with open('err_files.txt', 'a') as f:
                                f.write(oldimg + ':' + str(err) + '\n')
    print(num)
    specialfile.close()














