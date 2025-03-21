import os
import numpy as np
import cv2
import pyttsx3
# aiff文件转换成mp3编码文件模块
from pydub import AudioSegment

class Face:
    def __init__(self):
        self.datasetpath = './facedataset/'
        self.weights = './weight/train.yml'
        self.classifire_path = "./weight/haarcascade_frontalface_default.xml"
        self.facelabel = self.get_names(self.datasetpath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def load_weights(self):
        self.recognizer.read(self.weights)

    # 根据给定的人脸（x，y）坐标和宽度高度在图像上绘制矩形
    def draw_rectangle(self, img, rect):
        (x, y, w, h) = rect  # 矩形框
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 根据给定的人脸（x，y）坐标写出人名
    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)

    # 脸部检测函数
    def face_detect_demo(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(self.classifire_path)
        faces = face_detector.detectMultiScale(gray, 1.2, 6)
        # 如果未检测到面部，则返回原始图像
        if (len(faces) == 0):
            return None, None
        # 目前假设只有一张脸，xy为左上角坐标，wh为矩形的宽高
        (x, y, w, h) = faces[0]
        # 返回图像的脸部部分
        return gray[y:y + w, x:x + h], faces[0]

    def get_names(self, datasetpath):
        dir_names = os.listdir(datasetpath)
        return dir_names

    def ReFileName(self, dirPath):
        """
        :param dirPath: 文件夹路径
        :return:
        """
        # 对目录下的文件进行遍历
        faces = []
        for file in os.listdir(dirPath):
            # 判断是否是文件
            if os.path.isfile(os.path.join(dirPath, file)) == True:
                c = os.path.basename(file)
                name = dirPath + '\\' + c
                img = cv2.imread(name)
                # 检测脸部
                face, rect = self.face_detect_demo(img)
                # 我们忽略未检测到的脸部
                if face is not None:
                    # 将脸添加到脸部列表并添加相应的标签
                    # resize
                    face = cv2.resize(face, (320, 320))
                    faces.append(face)
        return faces

    def train(self):
        names = self.get_names(self.datasetpath)
        faces = []
        # 获取全部人脸
        for name in names:
            dir_path = self.datasetpath + name
            face = self.ReFileName(dir_path)
            faces.append(face)
        # 标签处理
        labels = []
        for index, face in enumerate(faces):
            label = np.array([index for i in range(len(face))])
            labels.append(label)
        # #拼接并打乱数据特征和标签
        x = np.concatenate(tuple(faces), axis=0)
        y = np.concatenate(tuple(labels), axis=0)

        index = [i for i in range(len(y))]  # test_data为测试数据
        np.random.seed(1)
        np.random.shuffle(index)  # 打乱索引
        train_data = x[index]
        train_label = y[index]
        # 分类器
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(train_data, train_label)
        # 保存训练数据
        recognizer.write(self.weights)

    def predict(self, image):
        # 生成图像的副本，保留原始图像
        img = image.copy()
        # 检测人脸区域
        face, rect = self.face_detect_demo(img)  # face_detect_demo前面的人脸检测函数
        # 预测人脸名字
        if type(face) != type(None):
            label = self.recognizer.predict(face)
            # print(label)#label[0]为名字，label[1]可信度数值越低，可信度越高（
            if label[1] <= 60:
                # 获取由人脸识别器返回的相应标签的人名
                label_text = self.facelabel[label[0]]
                # 在检测到的脸部周围画一个矩形
                self.draw_rectangle(img, rect)
                # 标出预测的人名
                self.draw_text(img, label_text, rect[0], rect[1])
                # 返回预测的图像
                return img, label[0]
            else:
                # 在检测到的脸部周围画一个矩形
                self.draw_rectangle(img, rect)
                # 标出预测的人名
                self.draw_text(img, "not find", rect[0], rect[1])
                # 返回预测的图像
                return img, None
        else:
            return img, None

    def face_detect(self, camera):
        # 执行预测
        # 模块初始化
        engine = pyttsx3.init()
        engine.say("请正视摄像头")
        engine.runAndWait()
        cap = cv2.VideoCapture(camera)
        assert (cap.isOpened() and "open camera failed")
        labels = []
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            pred_img, label = self.predict(frame)
            # print(label)
            if label != None:
                labels.append(label)
            cv2.imshow('result', pred_img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if len(labels) == 50:
                # print(labels)
                if len(set(labels[-10: -1])) == 1:
                    engine.say("完毕")
                    engine.runAndWait()
                    break
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
        cap.release()
        cv2.destroyAllWindows()
        return self.facelabel[labels[-1]]
