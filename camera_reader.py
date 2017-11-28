# -*- coding:utf-8 -*-
import cv2

from boss_train import Model
from image_show import show_image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    #cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = "./xml/haarcascade_frontalface_default.xml"
    model = Model()
    model.load()
    count = 0
    while True:
        _, frame = cap.read()

        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(100, 100))
        #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
        if len(facerect) == 1:
            #print('single face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:
                # 検出した顔を囲む矩形の作成
                #cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                #image = frame[y: y + height, x: x + width]

                result = model.predict(image)
                if result == 0:  # boss
                    print('Yes %4d' % (count))
                    count += 1
                    #show_image()
                else:
                    print('No')

        #10msecキー入力待ち
        k = cv2.waitKey(1000)
        #print('k = ', k)
        #Escキーを押されたら終了
        if k == 'q':
            break

    #キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()
