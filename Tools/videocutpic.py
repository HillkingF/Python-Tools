from PIL import Image
from pylab import *
import cv2
import os
# import xlwt

class videocheck():
    def __init__(self, filedir, newimgdir):
        self.dir = filedir
        self.imgdir = newimgdir
    def get_filesize(self):  #(M,兆)  获取视频大小
        file_byte = os.path.getsize(self.dir)
        return self.sizeConvert(file_byte)
    def sizeConvert(self, size):  # 单位换算
        K, M, G = 1024, 1024 ** 2, 1024 ** 3
        if size >= G:
            return str(size / G) + 'G Bytes'
        elif size >= M:
            return str(size / M) + 'M Bytes'
        elif size >= K:
            return str(size / K) + 'K Bytes'
        else:
            return str(size) + 'Bytes'
    def get_file_times(self):  #获取视频时长
        cap = cv2.VideoCapture(self.dir)
        if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
            # get方法参数按顺序对应下表（从0开始编号)
            rate = cap.get(cv2.CAP_PROP_FPS) # 帧速率
            frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频文件的帧数
            seconds = frame_number / rate
            cap.release()
            cv2.destroyAllWindows()
            return seconds, rate, frame_number   # 视频秒数， 速率， 视频文件的帧数
        else:
            cap.release()
            cv2.destroyAllWindows()
            return None
    def get_all_file(self):   #获取视频下所有的文件
        for root, dirs, files in os.walk(self.dir):
            return files  # 当前路径下所有非目录子文件
    def get_frame_img(self, allframe, startframe, finishframe):
        '''
        根据视频帧数和视频时长截取图片
        :return: no
        '''
        video_path = self.dir
        times = -1
        # 提取视频的频率，每25帧提取一个
        # frameFrequency = 25.0
        outPutDirName = self.imgdir  # 输出图片到当前目录vedio文件夹下
        if not os.path.exists(outPutDirName):  # 如果文件目录不存在则创建目录
            os.makedirs(outPutDirName)
        camera = cv2.VideoCapture(video_path)
        while True:
            times += 1  # 每一帧的序号
            res, image = camera.read()   # 读取每一帧,ret 返回值为true,当返回flase表示视频结束
            if not res:
                print('not res , not image')
                break
            else:
                print(outPutDirName + str(times) + '.jpg')
                cv2.imencode('.jpg', image)[1].tofile(outPutDirName + str(times) + '.jpg')
            # if times > startframe and times <= finishframe:
            #     print(outPutDirName + str(times) + '.jpg')
            #     cv2.imencode('.jpg', image)[1].tofile(outPutDirName + str(times) + '.jpg')
            #     # cv2.imwrite(outPutDirName + str(times) + '_night3.jpg', image)
            #     # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1) 读入中文命名
            #     # cv2.imencode('.jpg', img)[1].tofile(out_path)       # 写入含有中午名的图像
            # elif times > finishframe:
            #     break
        print('图片提取结束')
        camera.release()
        cv2.destroyAllWindows()

    def accoding_frame_toimg(self, videoname):
        # 要提取视频的文件名，隐藏后缀
        sourceFileName = videoname   #'a2'
        # 在这里把后缀接上
        video_path = os.path.join("", "", sourceFileName + '.mp4')
        times = 0
        # 提取视频的频率，每25帧提取一个
        frameFrequency = 30
        # 输出图片到当前目录vedio文件夹下
        outPutDirName = 'vedio/' + sourceFileName + '/'
        if not os.path.exists(outPutDirName):
            # 如果文件目录不存在则创建目录
            os.makedirs(outPutDirName)
        camera = cv2.VideoCapture(video_path)
        while True:
            times += 1
            res, image = camera.read()
            if not res:
                print('not res , not image')
                break
            if times % frameFrequency == 0:
                cv2.imwrite(outPutDirName + str(times) + '.jpg', image)
                print(outPutDirName + str(times) + '.jpg')
        print('图片提取结束')
        camera.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # 从视频中截取帧图像
    filepath = r'E:\data\Test\Car\12_9月8日_cy_路测车牌\路测车牌\privideo.mp4'
    newimgdir = r'E:\data\Test\Car\12_9月8日_cy_路测车牌\路测车牌\img'
    videoobj = videocheck(filepath, newimgdir)
    size = videoobj.get_filesize()   # 获取视频大小
    time, fvate, frame = videoobj.get_file_times()  # 获取时长：n秒  帧速率:一秒钟的帧数  帧数:=时长*帧速率
    # imgcount = int(frame/time)  # 根据帧数和视频时长计算截图数量
    print(time, fvate, frame)
    startframe = 0 * fvate   # 计算开始截图的帧数
    finishframe = 54 * fvate   # 计算结束截图的帧数
    videoobj.get_frame_img(frame, 0, 3344)
    print(size)

    exit()


    img = 'E:\\data\\Test\\Car\\car\\6cy发的第二批视频-测试相机清晰度\\192.168.10.252_01_2020060414451586_1\\4white\\'
    label = 'E:\\data\\Test\\Car\\car\\6cy发的第二批视频-测试相机清晰度\\192.168.10.252_01_2020060414451586_1\\4white.txt'
    for root, dir, files in os.walk(img):
        for file in files:
            hang = file + '\n'
            print(file)
            with open(label, 'a', encoding='utf-8')as f1:
                f1.write(hang)






'''
# cap = cv2.VideoCapture('C:\\Users\\Hillking\\Desktop\\192.168.1.64_01_20200417143253423.mp4')
# while(True):
#     ret,frame = cap.read()
#     cv2.imshow('face',frame)
#     cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()

# videoCapture = cv2.VideoCapture(r"C:\\Users\\Hillking\\Desktop\\192.168.1.64_01_20200417143253423.mp4")  # 捕捉视频，未开始读取；
# fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取视频帧速率
# size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸
#
# videoWrite = cv2.VideoWriter(r"D:\OpencvTest\zd1.avi",
#                              cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)
# cv2.VideoWriter_fourcc("I", "4", "2", "0")  .avi的未压缩的YUV颜色编码，文件较大
# cv2.VideoWriter_fourcc("P", "I", "M", "1")  .avi的MPEG-1编码类型
# cv2.VideoWriter_fourcc("X", "V", "I", "D")  .avi的MPEG-4编码类型
# cv2.VideoWriter_fourcc("T", "H", "E", "O")  .ogv的Ogg Vorbis
# cv2.VideoWriter_fourcc("F", "L", "V", "1")  .flv的flash视频

# t1 = cv2.getTickCount()  # CPU启动后总计数
#
# success, frame = videoCapture.read()  # 读帧
# while success:  # Loop until there are no more frames.
#     cv2.imshow("zd1", frame)
#     cv2.waitKey(int(1000 / fps))  # 1000毫秒/帧速率
#     videoWrite.write(frame)  # 写视频帧
#     success, frame = videoCapture.read()  # 获取下一帧
#
# t2 = cv2.getTickCount()
# print((t2 - t1) / cv2.getTickFrequency())  # cv2.getTickFrequency()返回CPU频率（每秒计数）


# # 用python把视频分解成图片
# # 读取一段视频
# cap=cv2.VideoCapture("d:/1.mp4")
# # 用作计数
# i=0
# # 循环判断视频是否打开
# while cap.isOpened():
#     # 读取每一帧,ret 返回值为true,当返回flase表示视频结束
#     ret, frame = cap.read()
#     # i=20 指定截取20张图片
#     if i == 20:
#         break
#     else:
#         i=i+1
#         # 图片命名及保存路径
#         filename="src"+str(i)+".jpg"
#         path='i:/result/{}'
#         # 保存图片
#         cv2.imwrite(path.format(filename),frame)
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()


#要提取视频的文件名，隐藏后缀
sourceFileName='a2'
#在这里把后缀接上
video_path = os.path.join("", "", sourceFileName+'.mp4')
times=0
#提取视频的频率，每25帧提取一个
frameFrequency=25
#输出图片到当前目录vedio文件夹下
outPutDirName='vedio/'+sourceFileName+'/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(video_path)
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
        print(outPutDirName + str(times)+'.jpg')
print('图片提取结束')
camera.release()
cv2.destroyAllWindows()
'''