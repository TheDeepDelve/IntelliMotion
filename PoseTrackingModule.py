import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smoothLandmarks=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smoothLandmarks,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        img = cv2.resize(img, (1000, 800))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
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
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        if len(self.lmList) <= max(p1, p2, p3):
            print("Insufficient landmarks detected")
            return None

        # Get coordinates for the three points
        x1, y1 = self.lmList[p1][1:]  # Shoulder
        x2, y2 = self.lmList[p2][1:]  # Elbow
        x3, y3 = self.lmList[p3][1:]  # Wrist

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle  # Normalize to 0–180 degrees

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
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        #print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Pose Detector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()