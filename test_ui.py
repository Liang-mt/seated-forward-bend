# -*- coding: utf-8 -*-

# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2

from test_angle import *
from face import *
from PIL import ImageFont, ImageDraw, Image
import serial
from yolov5 import *
import numpy as np
import pandas as pd



flag = 0
count = 0

def convert_cvimg_to_qtimg(cv_img):
    """将 OpenCV 图像转换为 Qt 图像"""
    rgb_frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    return pixmap
# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()

        self.setWindowTitle('坐位体前屈监测系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/xf.jpg"))
        # 图片读取进程
        self.output_size = 480
        # # 初始化视频读取线程
        self.vid_source = cam_id  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()

        self.detector = poseDetector()
        # 文本和绘制点的位置
        self.text = "腿弯曲过度，请保持规范姿势"
        self.engine = pyttsx3.init()




        self.initUI()
        self.reset_vid()

    '''
    ***模型初始化***
    '''


    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("坐位体前屈检测系统")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/sit.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)

        self.face_detection_btn = QPushButton("人脸识别")
        self.HW_detection_btn = QPushButton("身高体重监测")
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")

        self.face_detection_btn.setFont(font_main)
        self.HW_detection_btn.setFont(font_main)
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)

        self.face_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.HW_detection_btn.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(48,124,208)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")

        self.face_detection_btn.clicked.connect(self.open_facecam)
        self.HW_detection_btn.clicked.connect(self.open_HW)
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.face_detection_btn)
        vid_detection_layout.addWidget(self.HW_detection_btn)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用目标检测系统\n\n 提供付费指导：有需要的好兄弟加下面的QQ即可')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.addTab(vid_detection_widget, '视频检测')
        # self.addTab(about_widget, '联系我')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        # self.setTabIcon(2, QIcon('images/UI/lufei.png'))



    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()

            print(limst)
            df = pd.DataFrame([limst])  # 将字典转换为DataFrame对象
            write_dataframe_to_excel(df, 'limst.xlsx')  # 将 DataFrame 数据写入 Excel 文件并居中对齐
            # 强制退出程序
            QCoreApplication.instance().quit()


        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''
    def open_facecam(self):
        self.face_detection_btn.setEnabled(False)
        self.HW_detection_btn.setEnabled(False)
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = 0
        
        self.webcam = False
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.face_detect_vid)
        th.start()


    def open_HW(self):
        self.face_detection_btn.setEnabled(False)
        self.HW_detection_btn.setEnabled(False)
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = cam_id
        self.webcam = False
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.HW_detect_vid)
        th.start()

    def open_cam(self):
        self.face_detection_btn.setEnabled(False)
        self.HW_detection_btn.setEnabled(False)
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = cam_id
        self.webcam = False
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.face_detection_btn.setEnabled(False)
            self.HW_detection_btn.setEnabled(False)
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def face_detect_vid(self):
        output_size = self.output_size
        # 训练人脸
        face = Face()
        face.train()
        # 检测人脸
        face.load_weights()
        #模块初始化
        self.engine.say("现在开始人脸识别")
        self.engine.runAndWait()
        self.engine.say("请正视摄像头")
        self.engine.runAndWait()
        cap = cv2.VideoCapture(0)
        labels = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pred_img, label = face.predict(frame)
            if label is not None:
                labels.append(label)
            resize_scale = output_size / pred_img.shape[0]
            frame_resized = cv2.resize(pred_img, (0, 0), fx=resize_scale, fy=resize_scale)
            pixmap = convert_cvimg_to_qtimg(frame_resized)
            self.vid_img.setPixmap(pixmap)
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                break
            if len(labels) >= 8:
                
                if len(set(labels[-8:])) == 1:
                    self.engine.say("完毕")
                    self.engine.runAndWait()
                    break
        # 释放资源
        self.stopEvent.clear()
        self.face_detection_btn.setEnabled(True)
        self.HW_detection_btn.setEnabled(True)
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.reset_vid()

    def HW_detect_vid(self):
        weight = 0
        output_size = self.output_size
        self.engine.say("现在开始身高体重监测")
        self.engine.runAndWait()
        source = str(self.vid_source)
        if source == "0" or source == "1":
            source = int(source)
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        yolonet = yolov5(modelpath='./model/yolov5n.onnx', confThreshold=0.7, nmsThreshold=0.5, objThreshold=0.3)

        # 获取初始时间戳
        start_time = time.time()
        #print(start_time)

        while True:
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                break

                # 检测物体
            he, frame = yolonet.detect(frame)



            resize_scale = output_size / frame.shape[0]
            frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
            pixmap = convert_cvimg_to_qtimg(frame_resized)
            self.vid_img.setPixmap(pixmap)
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                break

            # 当前时间戳
            current_time = time.time()
            # 判断时间差是否超过20秒
            if current_time - start_time >= 10:
                if COM != "0":
                    weight = int(recv2(ser))
                    print(weight)
                    if weight > 0:
                        #print(current_time)
                        limst["身高"] = int(he / 373 * 163)
                        limst["体重"] = weight
                        #limst["体重"] = "65"
                        self.engine.say("完毕")
                        self.engine.runAndWait()
                        break

        # 释放资源
        self.stopEvent.clear()
        self.face_detection_btn.setEnabled(True)
        self.HW_detection_btn.setEnabled(True)
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.reset_vid()

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        global count
        output_size = self.output_size
        source = str(self.vid_source)
        if source == "0" or source == "1":
            source = int(source)
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        self.engine.say("现在开始坐位体前屈监测")
        self.engine.runAndWait()
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            # 获取视频帧的尺寸
            video_width = frame.shape[1]
            video_height = frame.shape[0]

            # 设置目标图像尺寸
            target_size = (video_width, video_height)

            # 将图像调整为目标尺寸
            frame = cv2.resize(frame, target_size)

            # 检测人体姿势
            frame =  self.detector.findPose(frame, draw=True)
            lmslist =  self.detector.findPosition(frame)

            current_time = time.time()
            if current_time - start_time >= 11:
                if COM != "0":
                    data = int(recv3(ser))
                    data = 38 - data
                    limst["距离"] = data
                    if limst["性别"] == "男":
                        limst["得分"] = caldis_boy(data)
                    if limst["性别"] == "女":
                        limst["得分"] = caldis_boy(data)
                    self.engine.say("完毕")
                    self.engine.runAndWait()
                    break

            if len(lmslist) == 0:
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                pixmap = convert_cvimg_to_qtimg(frame_resized)
                self.vid_img.setPixmap(pixmap)
                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    break
            else:

                # 获取右腿的角度
                right_angle =  self.detector.findAngle(frame, 24, 26, 28)
                point =  self.detector.Angledraw(frame, 26)
                point = (point[0] - 120, point[1] - 10)
                adjusted_point = adjust_point(point, frame.shape)
                left = adjusted_point[0]
                top = adjusted_point[1]
                if right_angle < 160:
                    count += 1
                    if count >= 5:
                        count = 5
                        frame = cv2ImgAddText(frame, self.text, left, top, (255, 0, 0), 20)
                else:
                    count = 0

                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                pixmap = convert_cvimg_to_qtimg(frame_resized)
                self.vid_img.setPixmap(pixmap)
                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    break

        self.stopEvent.clear()
        self.face_detection_btn.setEnabled(True)
        self.HW_detection_btn.setEnabled(True)
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.reset_vid()


    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.face_detection_btn.setEnabled(True)
        self.HW_detection_btn.setEnabled(True)
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/sit.jpeg"))
        self.vid_source = cam_id
        self.webcam = True
    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    limst = {}
    NAME = input("输入姓名：")
    limst["姓名"] = NAME
    SEX = input("输入性别：")
    limst["性别"] = SEX
    COM = input("请输入串口号：")
    if COM != "0":
        ser = serial.Serial(COM, 115200, timeout=1)
    cam_id = input("输入视频流地址：")
    # if cam_id == "0" or cam_id == "1":
    #     cam_id = int(cam_id)
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
