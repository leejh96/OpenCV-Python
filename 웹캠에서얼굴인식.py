import cv2
import numpy as np
font = cv2.FONT_ITALIC

def faceDetect():
    eye_detect = True
    faceCascade = cv2.CascadeClassifier(r"D:\OpenCV\FaceDetect-master\haarcascade_frontalface_default.xml")  # 얼굴 찾기 파일
    eyeCascade = cv2.CascadeClassifier(r"D:\OpenCV\FaceDetect-master\haarcascade_eye.xml")  # 눈 찾기 파일

    try:
        cap = cv2.VideoCapture(0)  # 0번 웹캠, 1번 USB 카메라
    except:
        print("camera loading error")
        return

    while True:
        retval, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 거울모드
        if not retval:
            break

        if eye_detect:
            info = "Eye Detention ON"
        else:
            info = "Eye Detention OFF"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = np.zeros(4)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)  # 카메라 영상위에 셋팅된 info 출력
        eyes = np.zeros((2, 4))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 범위
            if eye_detect:
                # 검출된 눈 영역의 컬러와 그레이스케일 관심 영역을 roi_color와 roi_gray에 저장
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eyeCascade.detectMultiScale(roi_gray)  # roi_gray에서 디폴트 (1.3, 5)로 눈 영역 검출
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(30)#바로 꺼지려면 지워야함

        if np.all(faces != 0) and np.all(eyes != 0):
            cv2.imwrite('data.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()
faceDetect()
