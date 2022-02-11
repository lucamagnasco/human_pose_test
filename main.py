import cv2
import time

import numpy as np

import pose_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Video:
    def __init__(self, video:str):
        """
        :param video: (str) path of video to analyse
        Object video has poseDetector inner or nested class, and a dataframe as attribute.
        """
        self.video=video
        self.df_results=pd.DataFrame()

    def run_analysis(self, show_video=True, plot=True, track_angle=None, track_heights=None):
        """
        :param show_video: (boolean) if True (def) shows video
        :param plot: (boolean) if True (def) plots scatterplot of [track_angle]
        :param track_angle: (ls) list of body joint to track angle of.
        :return: video and dataframe.
        """

        if track_angle is None:
            track_angle = ['right_knee', 'right_elbow']

        if track_heights is None:
            track_heights = ['right_foot', 'right_shoulder']

        #add columns to df_results
        for i in track_angle:
            self.df_results[f'{i}_angle'] = np.NAN
        for j in track_heights:
            self.df_results[f'{j}_height'] = np.NAN


        # Create video capture object:
        """for pc devices add number, for ex web cam real time video = 0"""
        cap = cv2.VideoCapture(self.video)
        pTime = 0

        # Create detector instance:
        self.detector = pm.poseDetector()

        # while True:
        while cap.isOpened():
            """ isOpened() method returns a boolean that indicates whether or not the video stream is valid"""
            frame, img = cap.read()
            """.read() method returns a tuple, where the first element is a boolean and the next element is the actual video frame. 
                 When the first element is True, it indicates the video stream contains a frame to read."""
            if frame == True:
                img = self.detector.findPose(img)

                # armo lista de listas con la posicion de cada pose_landmarks:
                lmlist = self.detector.findPosition(img)
                # list tiene Id para dif los landmarks --> si quiero el 14 traigo lmlist[14]

                # calcula frames by second (fps)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime  # ptime = 0 antes del while
                #imprime fps en imagen:
                cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)  #(255, 0, 0) es AZUL pq cv usa BGR

                if len(lmlist)!=0:
                    """lo bueno de hacerlo con objetos es que lo puedo hacer con cualquier punto"""
                    #todo: en vez de hardcodear se podria armar un diccionario
                    # todo: no hardcodear el agregado de las columnas al dataframe
                    # append unique value to results dataframe
                    if "right knee" in track_angle:
                        self.df_results.append({'right_knee_angle': self.detector.findAngle(img, 24,26,28)},
                                               ignore_index=True)
                    if "right arm" in track_angle:
                        self.df_results.append({'right_elbow_angle': self.detector.findAngle(img,  12, 14, 16)},
                                               ignore_index=True)

                    #add to results foot and shoulders heights log (in  lmlist[id][2] 2 is for cy) :

                    #todo:no me appendea los resultados
                    #right foot: id = 32 &
                    self.df_results.append({'right_foot_height': self.detector.lmlist[32][2]},
                                           ignore_index=True)
                    print(self.detector.lmlist[32][2], self.detector.lmlist[12][2])
                    # right shoulder: id = 12
                    self.df_results.append({'right_shoulder_height': self.detector.lmlist[12][2]},
                                           ignore_index=True)

                if show_video:
                    cv2.imshow("image", img)
                cv2.waitKey(120)

                #todo: agregar tipito con palitos solo sobre video
                #detector.create_landmark_graph()

                # si apreto la "q" mientras corre, cierra el codigo:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:      # else == si frame es false --> termina el video
                return self.df_results
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

        if plot:
            self.detector.scatter_log(self.df_results)

if __name__ == "__main__":
    video = Video('corte_cajon_igna_310.mp4')
    video.run_analysis()
    video.df_results


#TODO: observaciones: mejorar el seguimiento --> puntos inestables y al botin lo pierde ene l momento del impacto!
        ## aumentÉ trackcon
        ## aumentÉ waitKEy
#TODO: mediciones
# velocidad
# angulos
# alturas

#todo:comparar todas las patadas del mismo jugador para armar promedios --> mejorar
#todo: comparar contra jugadores profesional y sus numeros. --> conseguir video
#todo: object detection para detectar la pelota y visualizar cuando/donde (altura) impacta a la pelota.