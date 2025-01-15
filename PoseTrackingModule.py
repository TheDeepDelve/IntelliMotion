import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import math
import time

custom_model = load_model("pose_model.h5")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class IntegratedPoseDetector:
    def __init__(self, custom_model, mediapipe_model_complexity=1, mediapipe_detection_confidence=0.5):
        self.custom_model = custom_model
        self.mp_pose = mp_pose.Pose(
            model_complexity=mediapipe_model_complexity,
            min_detection_confidence=mediapipe_detection_confidence
        )

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)

    def predict_custom_model(self, img):
        preprocessed_img = self.preprocess_image(img)
        predictions = self.custom_model.predict(preprocessed_img)
        return predictions

    def find_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.mp_pose.process(img_rgb)
        return self.results

    def draw_pose(self, img, results, custom_predictions=None):
        if self.results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if custom_predictions is not None:
            for pred in custom_predictions:
                x, y = int(pred[0]), int(pred[1])
                cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        else:
            print("No landmarks detected.")
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        if len(self.lmList) <= max(p1, p2, p3):
            print("Insufficient landmarks detected")
            return None

        #coordinates for 3 points
        x1, y1 = self.lmList[p1][1:]  #shoulder
        x2, y2 = self.lmList[p2][1:]  #elbow
        x3, y3 = self.lmList[p3][1:]  #wrist

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle  #normalize to 0–180 degrees

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255,255), 3)
            cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,102), 2)

        return angle

def main():
    cap = cv2.VideoCapture(0)
    detector = IntegratedPoseDetector(custom_model)

    while True:
        success, img = cap.read()
        if not success:
            break

        #1: coarse keypoint detection using model
        custom_predictions = detector.predict_custom_model(img)
        custom_predictions = custom_predictions.reshape(-1, 2) * [img.shape[1], img.shape[0]]

        #2: fine-grained pose estimation using mediapipw
        mediapipe_results = detector.find_pose(img)

        #3: draw results from both models
        img = detector.draw_pose(img, mediapipe_results, custom_predictions)

        cv2.imshow("Integrated Pose Detector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()