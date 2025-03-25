import cv2
import numpy as np
import dlib
import pyautogui
import time
import mediapipe as mp
from collections import deque

pyautogui.FAILSAFE = False

# Konfigurasi layar
screen_width, screen_height = pyautogui.size()

# Load detektor wajah & landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Filter Kalman untuk smoothing
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.005

cap = cv2.VideoCapture(0)
time.sleep(2)

print("🔹 Silakan arahkan wajah ke tengah layar untuk kalibrasi selama 5 detik...")

# Data kalibrasi awal
calibration_data = {"nose_x": 0, "nose_y": 0}
calibrated = False

start_time = time.time()
samples = []

# Kalibrasi posisi kepala
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        nose_x, nose_y = landmarks.part(30).x, landmarks.part(30).y
        samples.append((nose_x, nose_y))

    cv2.imshow("Kalibrasi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if samples:
    calibration_data["nose_x"] = np.mean([s[0] for s in samples])
    calibration_data["nose_y"] = np.mean([s[1] for s in samples])
    calibrated = True
    print("✅ Kalibrasi selesai!")

cursor_x, cursor_y = screen_width // 2, screen_height // 2
pyautogui.moveTo(cursor_x, cursor_y)

buffer_size = 5
x_history = deque(maxlen=buffer_size)
y_history = deque(maxlen=buffer_size)

motion_threshold = 2.0
sensitivity = 1.2
use_hand_control = False
prev_hand_y = None  # Untuk scrolling yang lebih halus

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    faces = detector(gray)

    if faces and not use_hand_control:  # Gunakan kepala hanya jika tangan tidak aktif
        for face in faces:
            landmarks = predictor(gray, face)
            nose_x, nose_y = landmarks.part(30).x, landmarks.part(30).y

            if calibrated:
                delta_x = nose_x - calibration_data["nose_x"]
                delta_y = nose_y - calibration_data["nose_y"]

                head_x = np.interp(delta_x, [-40, 40], [0, screen_width])
                head_y = np.interp(delta_y, [-40, 40], [0, screen_height])

                # Gunakan Kalman Filter
                measurement = np.array([[np.float32(head_x)], [np.float32(head_y)]])
                kalman.correct(measurement)
                prediction = kalman.predict()
                new_x, new_y = int(prediction[0][0]), int(prediction[1][0])

                # Rata-rata bergerak
                x_history.append(new_x)
                y_history.append(new_y)

                smoothed_x = np.mean(x_history)
                smoothed_y = np.mean(y_history)

                if abs(smoothed_x - cursor_x) > motion_threshold or abs(smoothed_y - cursor_y) > motion_threshold:
                    cursor_x, cursor_y = smoothed_x, smoothed_y

                cursor_x = max(0, min(screen_width, cursor_x))
                cursor_y = max(0, min(screen_height, cursor_y))

                pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)

    # Deteksi tangan
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  # Jari telunjuk
            middle_tip = hand_landmarks.landmark[12]  # Jari tengah
            ring_tip = hand_landmarks.landmark[16]  # Jari manis
            pinky_tip = hand_landmarks.landmark[20]  # Kelingking
            thumb_tip = hand_landmarks.landmark[4]  # Ibu jari

            # Hitung jumlah jari yang terangkat
            fingers_up = sum([
                index_tip.y < hand_landmarks.landmark[6].y,
                middle_tip.y < hand_landmarks.landmark[10].y,
                ring_tip.y < hand_landmarks.landmark[14].y,
                pinky_tip.y < hand_landmarks.landmark[18].y
            ])

            hand_x, hand_y = int(hand_landmarks.landmark[0].x * screen_width), int(hand_landmarks.landmark[0].y * screen_height)

            # Jika tangan diangkat, gunakan tangan untuk mengontrol kursor
            if fingers_up > 0:
                use_hand_control = True
                cursor_x, cursor_y = hand_x, hand_y
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)

                # Scroll hanya jika 1 jari diangkat
                if fingers_up == 1:
                    if prev_hand_y is not None:
                        scroll_speed = int((prev_hand_y - hand_y) * 3)  # Faktor kecepatan
                        pyautogui.scroll(scroll_speed)
                    prev_hand_y = hand_y

                # Klik kanan jika 2 jari diangkat
                elif fingers_up == 2:
                    pyautogui.rightClick()

                # Klik kiri jika tangan mengepal (tidak ada jari terangkat)
                elif fingers_up == 0:
                    pyautogui.click()

            else:
                use_hand_control = False
                prev_hand_y = None

    cv2.imshow("Hand & Eye Tracking Cursor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
