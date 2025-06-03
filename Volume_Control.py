import cv2
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap=cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,1080)
pTime=0 

detector=htm.handDetector()

#volume access
devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        x1,y1=lmList[4][1],lmList[4][2] #x,y coordinate of thumb(4) nd index(8)
        x2,y2=lmList[8][1],lmList[8][2]
        cv2.circle(img,(x1,y1),15,(255,255,0),cv2.FILLED) #thumb
        cv2.circle(img,(x2,y2),15,(255,255,0),cv2.FILLED) #index
        cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3) #line b/w thumb nd index
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img, (cx, cy), 10, (255,255,0), cv2.FILLED) #circle at center of line

        #length b/w thumb nd index
        length=math.hypot(x2-x1,y2-y1)
        print(length)

        vol = np.interp(length, [20, 230], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        #red circle at 0% vol
        if length<30:
            cv2.circle(img, (cx, cy), 10, (25, 0, 255), cv2.FILLED)

        #grn circle at 100% vol
        if length>180:
            cv2.circle(img, (cx, cy), 10, (2, 113, 72), cv2.FILLED)

    cv2.imshow("Volume Control", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()