import cv2
import mediapipe as mp
import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import serial
import time
# -*- coding: utf-8 -*-

class poseDetector():

    def __init__(self, mode = False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        m_id = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                m_id.append(id)
                # finding height, width of the image printed
                h, w, c = img.shape
                # Determining the pixels of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        if 24 in m_id and 26 in m_id and 28 in m_id:
            return self.lmList
        else:
            return []

    def Angledraw(self,img,landmark):
        landpoint = (int(self.results.pose_landmarks.landmark[landmark].x * img.shape[1]),
                 int(self.results.pose_landmarks.landmark[landmark].y * img.shape[0]))
        return landpoint

    def checkLandmarks(self,x,y,z):
        landmarks_indices = [x, y, z]  # 关键点序号列表，这里假设需要检测的关键点序号为 11、12、23
        for index in landmarks_indices:
            if not hasattr(self.results.pose_landmarks.landmark[index], 'x') or not hasattr(
                    self.results.pose_landmarks.landmark[index], 'y'):
                return False
        return True

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate Angle
        if (x1 != x2) and (x3 != x2):
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                 math.atan2(y1 - y2, x1 - x2))
        else:
            return 999  # 如果两个点的x坐标相等，则无法计算角度

        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "./platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def adjust_point(point, image_shape, xmin=120, ymin=10, increment=5):
    x, y = point[0] - xmin, point[1] - ymin
    width, height = image_shape[1], image_shape[0]

    # 循环直到点完全符合条件
    while x < 0 or y < 0 or x >= width or y >= height:
        # 调整 x 坐标
        if x < 0:
            x += increment
        else:
            x -= increment

        # 调整 y 坐标
        if y < 0:
            y += increment
        else:
            y -= increment

    # 更新点的位置为调整后的值
    adjusted_point = (x + xmin, y + ymin)
    return adjusted_point




def recv2(serial):
    while True:
        data = serial.readline()
        if data:
            data = data.decode("utf-8")
            data = data[:3]  # 截取前两位数据
            break
        else:
            data = "123"  # 修改为字符串形式
            break
        sleep(0.02)
    return data

def recv3(serial):
    while True:
        data = serial.readline()
        if data:
            data = data.decode("utf-8")
            data = data[-4:-1]  # 截取后三位数据
            break
        else:
            data = "123"  # 修改为字符串形式
            break
        sleep(0.02)
    return data

def caldis_boy(x):
    if x < -9.0:
        y = 0
    elif x>=-9.0 and x<-8.0:
        y = 10
    elif x >= -8.0 and x < -7.0:
        y = 20
    elif x >= -7.0 and x < -6.0:
        y = 30
    elif x >= -6.0 and x < -5.0:
        y = 40
    elif x >= -5.0 and x < -4.0:
        y = 50
    elif x >= -4.0 and x < -2.7:
        y = 60
    elif x >= -2.7 and x < -1.4:
        y = 62
    elif x >= -1.4 and x < -0.1:
        y = 64
    elif x >= -1.4 and x < -0.1:
        y = 64
    elif x >= -0.1 and x < 1.2:
        y = 66
    elif x >= 1.2 and x < 2.5:
        y = 68


    elif x >= 2.5 and x < 3.8:
        y = 70
    elif x >= 3.8 and x < 5.1:
        y = 72
    elif x >= 5.1 and x < 6.4:
        y = 74
    elif x >= 6.4 and x < 7.7:
        y = 76
    elif x >= 7.7 and x < 9.0:
        y = 78
    elif x >= 9.0 and x < 11.5:
        y = 80

    elif x >= 11.5 and x < 14.0:
        y = 85
    elif x >= 14.0 and x < 15.3:
        y = 90
    elif x >= 15.3 and x < 16.6:
        y = 95
    elif x >= 16.6:
        y = 100
    return y

def caldis_girl(x):
    if x < -2.1:
        y = 0
    elif x>=-2.1 and x<-1.3:
        y = 10
    elif x >= -1.3 and x < -0.5:
        y = 20
    elif x >= -0.5 and x < 0.3:
        y = 30
    elif x >= 0.3 and x < 1.1:
        y = 40
    elif x >= 1.1 and x < 1.9:
        y = 50
    elif x >= 1.9 and x < 3.0:
        y = 60
    elif x >= 3.0 and x < 4.1:
        y = 62
    elif x >= 4.1 and x < 5.2:
        y = 64
    elif x >= 5.2 and x < 6.3:
        y = 66
    elif x >= 6.3 and x < 7.4:
        y = 68


    elif x >= 7.4 and x < 8.5:
        y = 70
    elif x >= 8.5 and x < 9.6:
        y = 72
    elif x >= 9.6 and x < 10.7:
        y = 74
    elif x >= 10.7 and x < 11.8:
        y = 76
    elif x >= 11.8 and x < 12.9:
        y = 78
    elif x >= 12.9 and x < 15.2:
        y = 80

    elif x >= 15.2 and x < 17.5:
        y = 85
    elif x >= 17.5 and x < 18.7:
        y = 90
    elif x >= 18.7 and x < 19.9:
        y = 95
    elif x >= 19.9:
        y = 100
    return y


if __name__ == "__main__":
    #ser = serial.Serial('COM3', 9600, timeout=1)

    # 文本和绘制点的位置
    text = "腿弯曲过度，请保持规范姿势"

    # 创建PoseDetector类对象
    detector = poseDetector()

    # 读取视频
    cap = cv2.VideoCapture("3.avi")

    # 连续小于160度的帧数
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 获取视频帧的尺寸
        video_width = frame.shape[1]
        video_height = frame.shape[0]

        # 设置目标图像尺寸
        target_size = (video_width, video_height)

        # 将图像调整为目标尺寸
        frame = cv2.resize(frame, target_size)

        # 检测人体姿势
        frame = detector.findPose(frame, draw=True)
        lmslist = detector.findPosition(frame)

        if len(lmslist) == 0:
            cv2.imshow('Video', frame)
            cv2.waitKey(30)
            continue

        #获取右腿的角度
        right_angle = detector.findAngle(frame, 24, 26, 28)
        point  = detector.Angledraw(frame, 26)
        point = (point[0]-120,point[1]-10)
        adjusted_point = adjust_point(point, frame.shape)
        print(adjusted_point)
        left = adjusted_point[0]
        top = adjusted_point[1]
        if right_angle == None:
            count = 0
        if right_angle < 160:
             count += 1
             if count > 5:
                count = 5
                frame = cv2ImgAddText(frame, text, left, top, (255, 0, 0), 20)
             else:
                frame = frame
        if right_angle >= 160:
             count = 0
        #data = int(recv(ser))
        # 将文本显示在窗口的左上角
        #dis = f'distance: {data}'
        #cv2.putText(frame, dis, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #print(count)
        cv2.imshow("Video", frame)
        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()