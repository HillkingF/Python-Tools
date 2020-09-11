# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QMessageBox, \
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import shutil

# http://code.py40.com/pyqt5/
class windows(QWidget):
    def __init__(self, imgdirpath, buquanpath, bufenpath):
        super().__init__()
        # 获得图片的名字列表和根路径
        self.imgdirpath = imgdirpath
        self.root, self.imgpaths = self.walkimgfile()  # 调用这个方法并获得返回值
        self.count = 0  # 用于判断执行了几次
        self.startornot = 0   # 用于判断是否开始
        self.buquanpath = buquanpath
        self.bufenpath = bufenpath

        QToolTip.setFont(QFont('SansSerif', 10))  # 静态方法设置一个用于提示的字体，这里使用10px大小的字体
        self.setToolTip('This is a <b>QWidget</b> widget')  # 创建一个提示，可以使用丰富的文本格式

        # next按钮控件
        startbtn = QPushButton('开始')   # 点击’开始‘后才能开始操作
        startbtn.clicked.connect(self.startButton)
        zhedangbtn = QPushButton('不全')  # 创建一个PushButton
        zhedangbtn.clicked.connect(self.firstButton)
        notzdbtn = QPushButton('部遮')
        notzdbtn.clicked.connect(self.notzdButton)

        # 设置布局,将按钮放在底部的水平布局中
        hbox = QHBoxLayout()   # 水平布局
        hbox.addWidget(startbtn)
        hbox.addWidget(zhedangbtn)
        hbox.addWidget(notzdbtn)

        # 文本框控件，用于计数
        self.numbertext = QLabel('完成数量', self)
        self.numbertext.setText('无')  # 默认完成了0个
        self.numbertext.setAlignment(Qt.AlignCenter)
        self.labelvbox = QVBoxLayout()
        self.labelvbox.addWidget(self.numbertext, 0, Qt.AlignCenter)
        # 图片格式转换
        path = "QLabel{border-image: url(" + "./background.jpg" + ");}"#./0_419900100337029842_1_118_K5_0_0_0.jpg'
        self.label11 = QLabel(self)
        self.label11.setToolTip('文本标签')
        self.label11.setStyleSheet(path)#path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
        self.label11.setFixedWidth(256)
        self.label11.setFixedHeight(48)
        self.imgvbox = QVBoxLayout()  # 这个小的布局用于存放图片
        self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)

        self.vbox = QVBoxLayout()   # 竖直布局
        self.vbox.addLayout(self.imgvbox)
        self.vbox.addLayout(self.labelvbox)

        # self.vbox.addStretch(1)     # 首先在竖直方向上插入留白空间
        self.vbox.addLayout(hbox)   # 插入上面的水平空间
        self.setLayout(self.vbox)   # 将竖直布局加入整体布局中

        # 窗口显示设置
        self.setGeometry(300, 300, 300, 150)  # 设置窗口的位置和大小
        self.setWindowTitle('数据处理器')  # 设置窗口的标题
        self.show()  # 显示窗口


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                '确认关闭并自动保存吗？',QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    def startButton(self):
        if self.startornot == 0:
            self.startornot = 1  # 表明可以开始了
            self.imgvbox.removeWidget(self.label11)  # 删去空白图
            if self.count - 1 < len(self.imgpaths):
                # 计数显示
                self.numbertext.setText('第 ' + str(1) + ' 张')
                # 图片切换
                self.label11 = QLabel(self)
                self.label11.setToolTip('文本标签')
                path = "QLabel{border-image: url(" + self.root.replace('\\', '/') + self.imgpaths[0] + ");}"  # ./0_419900100337029842_1_118_K5_0_0_0.jpg'
                self.label11.setStyleSheet(path)  # path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
                self.label11.setFixedWidth(256)
                self.label11.setFixedHeight(48)
                self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)

    def firstButton(self):
        if self.startornot == 1:
            # 先存储上一张图片
            shutil.copy(self.root + self.imgpaths[self.count], self.buquanpath + self.imgpaths[self.count])
            self.count += 1  # 开始查看下一张，从0开始
            self.imgvbox.removeWidget(self.label11)  # 删去上一张图片
            if (self.count-1 < len(self.imgpaths)) & (self.count != len(self.imgpaths)):
                # 计数显示
                self.numbertext.setText('第 ' + str(self.count) + ' 张')
                # 图片切换
                self.label11 = QLabel(self)
                self.label11.setToolTip('文本标签')
                path = "QLabel{border-image: url(" + self.root.replace('\\', '/') + self.imgpaths[self.count] + ");}"  # ./0_419900100337029842_1_118_K5_0_0_0.jpg'
                self.label11.setStyleSheet(path)  # path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
                self.label11.setFixedWidth(256)
                self.label11.setFixedHeight(48)
                self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)
                # 制作标签，写入txt
                # zdtxthang = self.imgpaths[self.count-1] + '\n'
                # with open('zhedang.txt', 'a', encoding='utf-8')as fzd:
                #     fzd.write(zdtxthang)
                #     print(self.imgpaths[self.count-1])
            elif self.count == len(self.imgpaths):
                # 计数显示
                self.numbertext.setText('Finish')
                # 图片切换
                self.label11 = QLabel(self)
                self.label11.setToolTip('文本标签')
                path = "QLabel{border-image: url(./background.jpg);}"  # ./0_419900100337029842_1_118_K5_0_0_0.jpg'
                self.label11.setStyleSheet(
                    path)  # path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
                self.label11.setFixedWidth(256)
                self.label11.setFixedHeight(48)
                self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)
                zdtxthang = self.imgpaths[self.count - 1] + '\n'
                with open('zhedang.txt', 'a', encoding='utf-8')as fzd:
                    fzd.write(zdtxthang)
                    print(self.imgpaths[self.count - 1])
                self.startornot = 2   # 表示 功能键也不管用了

    def notzdButton(self):
        if self.startornot == 1:
            shutil.copy(self.root + self.imgpaths[self.count], self.bufenpath + self.imgpaths[self.count])
            self.count += 1  # 开始查看下一张，从0开始
            self.imgvbox.removeWidget(self.label11)  # 删去上一张图片
            if (self.count - 1 < len(self.imgpaths)) & self.count != len(self.imgpaths):
                # 计数显示
                self.numbertext.setText('第 ' + str(self.count) + ' 张')
                # 图片切换
                self.label11 = QLabel(self)
                self.label11.setToolTip('文本标签')
                path = "QLabel{border-image: url(" + self.root.replace('\\', '/') + self.imgpaths[self.count] + ");}"  # ./0_419900100337029842_1_118_K5_0_0_0.jpg'
                self.label11.setStyleSheet(path)  # path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
                self.label11.setFixedWidth(256)
                self.label11.setFixedHeight(48)
                self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)
                # 制作标签，写入txt
                notzdtxthang = self.imgpaths[self.count - 1] + '\n'
                with open('notzhedang.txt', 'a', encoding='utf-8')as notfzd:
                    notfzd.write(notzdtxthang)
                    print(self.imgpaths[self.count - 1])
            else:
                # 计数显示
                self.numbertext.setText('Finish')
                # 图片切换
                self.label11 = QLabel(self)
                self.label11.setToolTip('文本标签')
                path = "QLabel{border-image: url(./background.jpg);}"  # ./0_419900100337029842_1_118_K5_0_0_0.jpg'
                self.label11.setStyleSheet(path)  # path = "QLabel{border-image: url(./0_419900100337029842_1_118_K5_0_0_0.jpg);}"
                self.label11.setFixedWidth(256)
                self.label11.setFixedHeight(48)
                self.imgvbox.addWidget(self.label11, 0, Qt.AlignCenter)
                notzdtxthang = self.imgpaths[self.count - 1] + '\n'
                with open('notzhedang.txt', 'a', encoding='utf-8')as notfzd:
                    notfzd.write(notzdtxthang)
                    print(self.imgpaths[self.count - 1])
                self.startornot = 2



    def walkimgfile(self):
        for root, dirs,files in os.walk(self.imgdirpath):
            pass
        return root, files


if __name__ == '__main__':
    priimgdir = 'C:\\Users\\office\\Desktop\\cover的测试结果-副本\\restv0\\'
    buquanpath = 'E:\\data\\carplat\\car_dirty_cover\\car_cover\\cover的测试结果\\v1_结果分类\\剪切不全\\'
    bufenpath = 'E:\\data\\carplat\\car_dirty_cover\\car_cover\\cover的测试结果\\v1_结果分类\\部分遮挡\\'
    app = QApplication(sys.argv)
    usetool = windows(priimgdir, buquanpath, bufenpath)
    sys.exit(app.exec_())


