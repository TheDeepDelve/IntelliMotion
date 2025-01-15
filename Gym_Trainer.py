import cv2
import numpy as np
import time
from ComputerVisionProject import PoseTrackingModule as pm

#cap = cv2.VideoCapture("Gym_Videos/1.mp4")
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
dir = 0  #0 for going up and 1 for going down

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 12, 14, 16)
        percentage = np.interp(angle, (40, 160), (0, 100))
        bar = np.interp(angle, (40, 160), (400, 100))

        if percentage >= 100:
            bar_color = (0, 255, 0)#grn
        else:
            bar_color = (0, 0, 255)#red

        cv2.rectangle(img, (50, 100), (85, 400), bar_color, 3)
        cv2.rectangle(img, (50, int(bar)), (85, 400),bar_color, cv2.FILLED)
        cv2.putText(img, f'{int(percentage)}%', (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        #counting curls
        if percentage <= 5:  #top of curl
            if dir == 0:
                count += 0.5
                dir = 1
        if percentage >= 95:  #bottom of curl
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.putText(img, f'Count: {int(count)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("AI Trainer", img)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()