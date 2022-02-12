import pandas as pd
import cv2
import mediapipe as mp
import math

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.3, trackCon=0.99):
        self.mp_drawing = None
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
                                     model_complexity=2,
                                     smooth_segmentation = self.smooth,
                                     min_detection_confidence = self.detectionCon, # if confid > detectionCon --> go to tracking
                                     min_tracking_confidence = self.trackCon) # if track < trackCon --> go back to detection
        # mediapipe id landmark dictionary (picture)
        self.pose_lm_dict = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip':23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_foot': 31,
            'right_foot': 32
        }

    def findPose(self, img, draw=True):
        # mp needs RGB (cv2 operates with BGR) so I convert img:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # assign results to object:
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:  # dibujar puntos y lineas
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
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

    """z: Represents the landmark depth with the depth at the midpoint of hips being 
    the origin, and the smaller the value the closer the landmark is to the camera. 
    The magnitude of z uses roughly the same scale as x"""

    def findAngle(self, img, p1, p2, p3, draw=True):
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

    def create_landmark_graph(self):
        """
        Plot pose world landmarks.
        """
        self.mpDraw.plot_landmarks(self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)

    # todo: crear def para graficar seguimiento de angulos a partir del dataframe.
    def scatter_log(self, df):
        # df = main('corte_cajon_igna_310.mp4')
        # sns.scatterplot(data=df, x='index', y='angle')
        # plt.show()
        pass