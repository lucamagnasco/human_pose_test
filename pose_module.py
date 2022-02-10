import pandas as pd
import cv2
import mediapipe as mp
import math

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.3, trackCon=0.99):
        self.mode = mode
        self.smooth = smooth
        self.upBody = upBody
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.log_angle = []

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode, #if True always trys to find new detections
                                     # upBody = self.upBody, # detect only upper body
                                     smooth_segmentation = self.smooth,
                                     min_detection_confidence = self.detectionCon, # if confid > detectionCon --> go to tracking
                                     min_tracking_confidence = self.trackCon) # if track < trackCon --> go back to detection
        # self.points = mp.PoseLandmark
        # self.point_mov_hist = []

        # for p in self.points:
        # print(p)
        # x = str(p)[13:]
        # self.point_mov_hist.append(x + "_z")
        # self.point_mov_hist.append(x + "_y")
        # self.point_mov_hist.append(x + "_x")
        # self.point_mov_hist.append(x + "_vis")

        # self.df = pd.DataFrame(columns=self.point_mov_hist)

    def findPose(self, img, draw=True):
        # mp needs RGB (cv2 operates with BGR) so I convert img:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # assign results to object:
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:  # dibujar puntos y lineas
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            # landmarks = self.results.pose_landmarks.landmark
            #
            # for i, j in zip(self.points, landmarks):
            #     temp = temp + [j.x, j.y, j.z, j.visibility]
            #
            # count = 0
            # self.point_mov_hist.loc[count] = temp
            # count += 1
            # self.point_mov_hist.to_csv("point_mov_hist.csv")
        return img

    def findPosition(self, img, draw=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                """
                id: landmark id, 
                lm: landmark info (normalized) """

                # img shape = (height, width, channel)
                h, w, c = img.shape

                # To get the actual pixel values from lm ratios, we multiply X and y by w and h of img, respectively.
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])  # , lm.z]) #todo: agregar Z profundidad
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmlist

    def findeAngle(self, img, p1, p2, p3, draw=True):
        """encontrar angulo entre p1, p2, p3"""
        # Get the landmarks
        x1, y1 = self.lmlist[p1][1:]  # agarrame del p1 toda la lista ignorando el primer elemento(id)
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        # calculate angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        # para evitar angulo neg:
        #todo: mejorar esto
        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 80, y2 + 5),      # -80 & -5 es para que tape el punto el ang
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)  # formato

        return int(angle)
