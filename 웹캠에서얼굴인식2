{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 라이브러리"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "import face_recognition\n",
    "import cv2\n",
    "import numpy as np\n",
    "import glob\n",
    "import dlib\n",
    "import time\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 함수"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "font = cv2.FONT_HERSHEY_DUPLEX\n",
    "try:\n",
    "    cap = cv2.VideoCapture(0) # 0번 웹캠, 1번 USB 카메라\n",
    "except:\n",
    "    print(\"camera loading error\")\n",
    "\n",
    "def ageGenderPredict():\n",
    "    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']\n",
    "    \n",
    "    age_net = cv2.dnn.readNetFromCaffe(\n",
    "              'D:/python/age_gender_estimation-master/models/deploy_age.prototxt', \n",
    "              'D:/python/age_gender_estimation-master/models/age_net.caffemodel')\n",
    "\n",
    "    \n",
    "    img1 = face_recognition.load_image_file(\"D:/python/data//img1.jpg\")\n",
    "    img1_face_encoding = face_recognition.face_encodings(img1)[0]\n",
    "    img2 = face_recognition.load_image_file(\"D:/python/data//img2.jpg\")\n",
    "    img2_face_encoding = face_recognition.face_encodings(img2)[0]\n",
    "    img3 = face_recognition.load_image_file(\"D:/python/data//img3.jpg\")\n",
    "    img3_face_encoding = face_recognition.face_encodings(img3)[0]\n",
    "    \n",
    "    known_face_encodings = [\n",
    "        img1_face_encoding,\n",
    "        img2_face_encoding,\n",
    "        img3_face_encoding,\n",
    "    ]\n",
    "\n",
    "    # 미리 학습시킨 이미지의 이름을 추가\n",
    "    known_face_names = [\n",
    "        \"LeeJuHyuk\",\n",
    "        \"ParkByeongHyun\",\n",
    "        \"HeeGyeong\",\n",
    "    ]\n",
    "    \n",
    "    face_locations = []#위,아래,좌,우\n",
    "    face_encodings = []# 벡터모음\n",
    "    face_name = []#csv에 넣을 이름\n",
    "    face_age = []#csv에 넣을 나이\n",
    "    post_data = []\n",
    "    \n",
    "    while True:\n",
    "        ret, frame = cap.read()\n",
    "        frame = cv2.flip(frame, 1)#거울모드\n",
    "        #빠른 인식을 위해 얼굴부분을 감지하는 사각형을 1/4로 축소(얼굴크기)\n",
    "        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)\n",
    "\n",
    "        # face_recognition의 인수로 사용하기 위해 BGR 이미지를 RGB로 변경\n",
    "        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "\n",
    "        # 초기의 배열에 location값과 encoding 값 넣기\n",
    "\n",
    "        face_locations = face_recognition.face_locations(rgb_small_frame)\n",
    "        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)\n",
    "\n",
    "        \n",
    "        for face_encoding in face_encodings:\n",
    "            #얼굴이 이미 학습된 얼굴이미지와 같은지 확인\n",
    "            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.4)#T/F\n",
    "            name = \"Unknown\"\n",
    "            #distance가 가장 적은 이미지를 찾아서 이름을 출력(0.6정도가 좋다고 함)\n",
    "            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)\n",
    "            best_match_index = np.argmin(face_distances)#argmin 최솟값 인덱스찾는 속성 1,0\n",
    "            if matches[best_match_index] == True:\n",
    "                name = known_face_names[best_match_index]\n",
    "            face_name.append(name)\n",
    "\n",
    "            # 두개의 배열을 합치는데 인덱스가 같은 것끼리 합치는 것을 zip\n",
    "        for (top, right, bottom, left) in face_locations:\n",
    "            #탐지된 크기가 1/4이였으므로 4배를 다시 키워줌\n",
    "            top *= 4\n",
    "            right *= 4\n",
    "            bottom *= 4\n",
    "            left *= 4\n",
    "\n",
    "            # 얼굴 감지하는 사각형 그리기\n",
    "            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)\n",
    "\n",
    "            # 사각형 밑에 이름 쓰기\n",
    "            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)\n",
    "            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)\n",
    "\n",
    "        #(top, right, bottom, left)\n",
    "        #face_locations = face_recognition.face_locations(rgb_small_frame)\n",
    "        #x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()\n",
    "        for (top, right, bottom, left) in face_locations:\n",
    "            face_img = frame[top:bottom, left:right]\n",
    "            blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),\n",
    "                    mean=(78.4263377603, 87.7689143744, 114.895847746),\n",
    "                    swapRB=False, crop=False)\n",
    "\n",
    "            # predict age\n",
    "            age_net.setInput(blob)\n",
    "            age_preds = age_net.forward()\n",
    "            age = age_list[age_preds[0].argmax()]\n",
    "            face_age.append(age)\n",
    "            \n",
    "            # visualize\n",
    "            overlay_text = '%s' % (age)\n",
    "            cv2.putText(frame, overlay_text, org=(left, top),\n",
    "              fontFace=font, fontScale=1, color=(255,255,255), thickness=2)\n",
    "        \n",
    "        now = time.strftime(\"%Y_%m_%d_%H_%M_%S\")\n",
    "\n",
    "        cv2.imwrite('D:/python/data/'+now+'.jpg',frame)\n",
    "        cv2.imshow('Video', frame)\n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "        \n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()\n",
    "    \n",
    "    for face_name, age in zip(face_name, face_age):\n",
    "        post_data.append((face_name, age))\n",
    "    dataframe = pd.DataFrame(post_data)\n",
    "    dataframe.to_csv(\"D:/python/data/result/\" + now + \"postdata.csv\",header=False, index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 실행"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "ageGenderPredict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
