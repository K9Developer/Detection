import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=0)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img , 1)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)