import cv2
import numpy as np
import dlib

# Load detector wajah
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Unduh dari http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Ambil area mata dari landmark wajah
def get_eye_region(landmarks, eye_points):
    x = [landmarks.part(i).x for i in eye_points]
    y = [landmarks.part(i).y for i in eye_points]
    return min(x), min(y), max(x), max(y)

# Proses deteksi pupil
def detect_pupil(eye_roi):
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)  # Ubah nilai threshold jika tidak akurat
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        return int(cx), int(cy), int(radius)
    return None

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Ambil area mata kiri & kanan
        left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])

        # Potong area mata dari gambar
        left_eye_roi = frame[left_eye_region[1]:left_eye_region[3], left_eye_region[0]:left_eye_region[2]]
        right_eye_roi = frame[right_eye_region[1]:right_eye_region[3], right_eye_region[0]:right_eye_region[2]]

        # Deteksi pupil
        left_pupil = detect_pupil(left_eye_roi)
        right_pupil = detect_pupil(right_eye_roi)

        # Gambar pupil di area mata
        if left_pupil:
            cv2.circle(left_eye_roi, (left_pupil[0], left_pupil[1]), left_pupil[2], (0, 255, 0), 2)
        if right_pupil:
            cv2.circle(right_eye_roi, (right_pupil[0], right_pupil[1]), right_pupil[2], (0, 255, 0), 2)

        # Gambar kotak di area mata
        cv2.rectangle(frame, (left_eye_region[0], left_eye_region[1]), (left_eye_region[2], left_eye_region[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (right_eye_region[0], right_eye_region[1]), (right_eye_region[2], right_eye_region[3]), (255, 0, 0), 2)

    cv2.imshow("Pupil Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
