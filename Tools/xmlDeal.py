import os
import cv2
import numpy as np
import random
from xml.dom.minidom import parse



class xmlDeal:
    def __init__(self):
        self.numbers = []  # 车牌号 ObjectText
        self.colors = []  # 单双排字符串
        self.coordinates = []  # 坐标
        self.rtn = []  # 最后返回的列表
        self.panbies = []  # 暂时没用到
        self.count = 0

    # 根据四个角点裁剪图片  空间变换使用的方法是透射变换
    def cutpic(self, oldpath, zuobiao, sizechange):
        try:
            # 1、获取图像
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

            # 3、透视变换的关键代码。 pts1中坐标顺序是 左上、右上、右下、左下， 必须与pts2对应
            pts1 = np.array(
                [[float(zuobiao[0][0].split('.')[0]) - zuox, float(zuobiao[0][1].split('.')[0]) - shangy],  # 左上
                 [float(zuobiao[1][0].split('.')[0]) + youx, float(zuobiao[1][1].split('.')[0]) - shangy],  # 右上
                 [float(zuobiao[2][0].split('.')[0]) + youx, float(zuobiao[2][1].split('.')[0]) + xiay],  # 右下
                 [float(zuobiao[3][0].split('.')[0]) - zuox, float(zuobiao[3][1].split('.')[0]) + xiay]],  # 左下
                dtype='float32')
            pts2 = np.array([[0, 0], [256, 0], [256, 48], [0, 48]], dtype='float32')
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img, M, (256, 48))

            return warped
            # cv2.imencode('.jpg', warped)[1].tofile(newpath)   不在这里保存，因为还要根据类型标签来分类

        except Exception as err:
            return 'cut_error'

    # 使用minidom解析器打开xml文档
    def xmlAnalyse(self, xmlpath):
        DOMTree = parse(xmlpath)
        collection = DOMTree.documentElement

        # 规定： 遇到'MOAKey'开始记录、解析
        moakeys = collection.getElementsByTagName('MOAKey')
        for moakey in moakeys:   # 获取每一段moakeys的内容
            zuobiao = []

            ## 解析车牌号，确定车牌判别类型
            try:  # objectText有车牌号
                number = moakey.getElementsByTagName('ObjectText')[0].childNodes[0].data.replace(' ', '')
                panbie = '2'
            except Exception as err:
                number = '无'
                panbie = '2'

            ## 解析车牌颜色
            lable = moakey.getElementsByTagName('Lable')[0]
            try:
                color = lable.getElementsByTagName('Text')[0].childNodes[0].data
            except Exception as err:
                color = '无色层'

            ## 解析检测框坐标   这里有一些问题，没有判断坐标顺序，可能出现问题。需要根据x，y比较一下   需改进
            points = moakey.getElementsByTagName('Points')[0]  # moakey获取的是一个数组，0表示数组中的第0个元素
            pointn = points.getElementsByTagName('Point')  # 获取Points中所有的Point
            for point in pointn:
                x = point.getElementsByTagName('X')[0].childNodes[0].data
                y = point.getElementsByTagName('Y')[0].childNodes[0].data
                zuobiao.append([x, y])
            if len(zuobiao) == 5:
                self.coordinates.append(zuobiao)  # zuobiao  一组一定有五个
                self.numbers.append(number)
                self.panbies.append(panbie)
                self.colors.append(color)
                self.count += 1


        self.rtn.append(self.coordinates)
        self.rtn.append(self.numbers)
        self.rtn.append(self.panbies)
        self.rtn.append(self.colors)
        self.rtn.append(self.count)
        return self.rtn




count = -1
num = 0
tool = xmlDeal()

priimgdir = 'E:/data/motorcycle/20200728摩托车车牌13081张图+17985个车牌/'
oa = 'E:/data/motorcycle/oa/oa/'
cutimg = 'E:/data/motorcycle/cutimg/'
newtxt = 'E:/data/motorcycle/'

# 字典
letters_lower = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新使ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警领港澳学挂'
newdic = dict({x: i for i, x in enumerate(list(letters_lower))})

for root, dirs, filenames in os.walk(oa):
    for filename in filenames:
        if filename.endswith('.oa'):  # 判断是否是oa文件
            # 解析xml： ①原始图片  新图片  路径； ②到xml中获取坐标信息 ③④⑤返回值分别是坐标组、id组、type组
            xml = oa + filename
            rtn = tool.xmlAnalyse(xml)  # 返回值分别是：rtn[坐标组，id组，type组，车牌号组
            zuobiaos = rtn[0]
            number = rtn[1]
            panbie = rtn[2]
            color = rtn[3]
            jccount = rtn[4]  # 这一个是数字：个数，其余4个都是列表

            if jccount == 0:  # 说明这个xml中没有检测到的车牌
                continue
            else:
                for i, coordinates in enumerate(zuobiaos):
                    try:  # 对一组点开始进行变换_处理一个车牌
                        oldjpg = priimgdir + filename.replace('.oa', '.jpg')  # 加.是为了避免图片名中也含有字符oa
                        oldjpeg = priimgdir + filename.replace('.oa', '.jpeg')
                        oldimg = [oldjpg, oldjpeg]
                        if os.path.exists(oldimg[0]) or os.path.exists(oldimg[1]):
                            count += 1
                            # 若车牌号中有不是字典中的字符，则添加到special目录下
                            carsign = True
                            if number[i] == '无':
                                carsign = False
                            for x in number[i]:
                                if (x in newdic.keys()) and (len(number) != 0):
                                    pass
                                else:
                                    carsign = False

                            # 解析颜色 color
                            dic_color = {
                                'table1yellow': '3',
                                'table2yellow': '33',
                                'table1blue': '1',
                                'table2blue': '11',
                                'table1white': '2',
                                'table2white': '22',
                            }

                            # 根据信号选择切图保存之处
                            if carsign == True:  # 车牌号都是字典中的
                                # 原始尺寸的剪切填写
                                normalcutres = tool.cutpic(oldimg, coordinates, sizechange=False)
                                if normalcutres is not 'cut_error':
                                    normalhang = str(count) + '_' + number[i] + '.jpg'  # 图片名
                                    numbertype = []
                                    for x in number[i]:
                                        numbertype.append(newdic[x])
                                    for x in numbertype:
                                        normalhang = normalhang + ' ' + str(x)

                                    normalhang = normalhang + ' ' + dic_color[color[i]] + ' ' + panbie[i] + '\n'
                                    print(filename)
                                    print(normalhang)
                                    cv2.imencode('.jpg', normalcutres)[1].tofile(
                                        cutimg + str(count) + '_' + number[i] + '.jpg')
                                    txt = newtxt + 'label.txt'
                                    with open(txt, 'a', encoding='utf-8')as f1:
                                        f1.write(filename + '\n')
                                        f1.write(normalhang)
                            else:  # 车牌号不在字典中
                                pass

                        else:
                            with open('nopicture.txt', 'a') as f:
                                f.write(filename + '\n')

                    except Exception as err:
                        print('发生错误！ ' + oldimg + ':' + str(err))
                        with open('err_files.txt', 'a') as f:
                            f.write(oldimg + ':' + str(err) + '\n')