import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist

# Calculate the eye aspect ratio to detect eye closure or blink
def eye_aspect_ratio(eye):
    # Computes the euclidean distance between sets of vertical and horizontal eye landmark points
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # Returns the eye aspect ratio
    return (A + B) / (2.0 * C)

# Threshold values for determining engagement based on eye aspect ratio
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Load pre-trained models for face detection and landmark prediction
net = cv2.dnn.readNetFromCaffe('/users/andrewvaz/Documents/deploy.prototxt', '/users/andrewvaz/Documents/res10_300x300_ssd_iter_140000.caffemodel')
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("/users/andrewvaz/Downloads/shape_predictor_68_face_landmarks.dat")

# Initialize the video capture for real-time video processing
cap = cv2.VideoCapture(0)

# Initialize counters for real-time engagement analysis
engaged_count = 0
disengaged_count = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            dlib_rect = dlib.rectangle(startX, startY, endX, endY)
            landmarks = dlib_predictor(frame, dlib_rect)

            leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Increment counters based on eye aspect ratio
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Disengaged", (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                disengaged_count += 1
            else:
                engaged_count += 1

            for n in range(0, 68):
                cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 1, (0, 255, 255), -1)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    total_frames += 1
    engagement_score = (engaged_count / total_frames) * 100 if total_frames > 0 else 0
    cv2.putText(frame, f"Engagement: {engagement_score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('FaceTime', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
