from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import playsound
import time
from threading import Thread
import dlib
import os
import cv2

# Cau hinh duong dan den file alarm.wav
wav_path = "/home/pi/miai/landmark/alarm.wav"

# Ham phat ra am thanh
def play_sound(path):
	os.system('aplay ' + path)

# Ham tinh khoang cach giua 2 diem
def e_dist(pA, pB):
	return np.linalg.norm(pA - pB)

def eye_ratio(eye):
	# Tinh toan khoang cach theo chieu doc giua mi tren va mi duoi
	d_V1 = e_dist(eye[1], eye[5])
	d_V2 = e_dist(eye[2], eye[4])

	# Tinh toan khoang cach theo chieu ngang giua 2 duoi mat
	d_H = e_dist(eye[0], eye[3])

	# Tinh ty le giua chieu doc va chieu ngang
	eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)

	return eye_ratio_val

# Dinh nghia muc threshold ty le giua doc/ngang. Neu duoi muc nay la ngu gat
eye_ratio_threshold = 0.25

# Threshold so frame lien tuc nham mat
max_sleep_frames = 16

# Dem so frame ngu
sleep_frames = 0

# Check xem da canh bao hay chua
alarmed = False

# Khoi tao cac module detect mat va facial landmark
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Lay danh sach cac cum diem landmark cho 2 mat
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Doc tu camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:

	# Doc tu camera
	frame = vs.read()

	# Resize de tang toc do xu ly
	frame = imutils.resize(frame, width=450)

	# Chuyen ve gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect cac mat trong anh
	faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100),		flags=cv2.CASCADE_SCALE_IMAGE)

	# Duyet qua cac mat
	for (x, y, w, h) in faces:

		# Tao mot hinh chu nhat quanh khuon mat
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		# Nhan dien cac diem landmark
		landmark = landmark_detect(gray, rect)
		landmark = face_utils.shape_to_np(landmark)

		# Tinh toan ty le mat phai va trai va trung binh cong 2 ratio
		leftEye = landmark[left_eye_start:left_eye_end]
		rightEye = landmark[right_eye_start:right_eye_end]

		left_eye_ratio = eye_ratio(leftEye)
		right_eye_ratio = eye_ratio(rightEye)

		eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

		# Ve duong bao quanh mat
		left_eye_bound = cv2.convexHull(leftEye)
		right_eye_bound = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)

		# Check xem mat co nham khong
		if eye_avg_ratio < eye_ratio_threshold:
			sleep_frames += 1
			# if the eyes were closed for a sufficient number of
			# frames, then sound the alarm
			if sleep_frames >= max_sleep_frames:

				if not alarmed:
					alarmed = True
					# Duong dan den file wav


					# Tien hanh phat am thanh trong 1 luong rieng
					t = Thread(target=play_sound,
							   args=(wav_path,))
					t.deamon = True
					t.start()

				# Ve dong chu canh bao
				cv2.putText(frame, "BUON NGU THI DI NGU DI ONG OI!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Neu khong nham mat thi
		else:

			# Reset lai cac tham so
			sleep_frames = 0
			alarmed = False

			# Hien thi gia tri eye ratio trung binh
			cv2.putText(frame, "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio), (10, 30),	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

	# Hien thi len man hinh
	cv2.imshow("Camera", frame)

	# Bam Esc de thoat
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break


cv2.destroyAllWindows()
vs.stop()