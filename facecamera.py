import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置计数器初始值
counter = 1

# 循环读取摄像头画面
while True:
    # 读取画面
    ret, frame = cap.read()

    # 显示画面
    cv2.imshow("Camera", frame)

    # 检测按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 "q" 键，保存图片并退出循环
    if key == ord('q'):
        # 生成照片名称
        filename = "../test_ui/facedataset/name/{}.jpg".format(counter)

        # 保存图片
        cv2.imwrite(filename, frame)

        # 输出图片路径
        print("图片已保存至: ", filename)

        # 增加计数器
        counter += 1

    # 如果按下 "esc" 键，退出循环
    if key == 27:
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()