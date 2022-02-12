import cv2
import time
import numpy as np
import pose_module as pm
import pandas as pd


class Video:
    def __init__(self, video_path: str):
        """
        :param video_path: path of video to analyse
        Object video has poseDetector inner or nested class, and a dataframe as attribute.
        """
        self.df_results = pd.DataFrame()
        self.video = video_path

    def run_analysis(self, show_video=True, plot=True, track_angle=None, track_heights=None):
        """
        :param show_video: (boolean) if True (def) shows video
        :param plot: (boolean) if True (def) plots scatterplot of [track_angle]
        :param track_angle: (ls) list of body joint to track angle of.
        :param track_heights:  (ls) list of body joint to track heights of.
        :return: video and dataframe.
        """

        # Create detector instance:
        self.detector = pm.poseDetector()
        valid_body_string = self.detector.pose_lm_dict.keys()

        if track_angle is None:
            track_angle = ['right_knee', 'right_elbow']

        if track_heights is None:
            track_heights = ['right_foot', 'right_shoulder']

        # add columns to df_results
        for i in track_angle:
            self.df_results[f'{i}_angle'] = np.NAN
        for j in track_heights:
            self.df_results[f'{j}_height'] = np.NAN

        # Create video capture object:
        """for pc devices add number, for ex web cam real time video = 0"""
        cap = cv2.VideoCapture(self.video)
        pTime = 0

        # while True:
        while cap.isOpened():
            """ isOpened() method returns a boolean that indicates whether or not the video stream is valid"""
            frame, img = cap.read()
            """.read() method returns a tuple, where the first element is a boolean and the next element is the actual video frame. 
                 When the first element is True, it indicates the video stream contains a frame to read."""
            if frame:
                img = self.detector.findPose(img)

                # armo lista de listas con la posicion de cada pose_landmarks
                lmlist = self.detector.findPosition(img)

                # calcula frames by second (fps)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime  # ptime = 0 antes del while
                # imprime fps en imagen:
                cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 0, 0), 3)

                if len(lmlist) != 0:
                    """lo bueno de hacerlo con objetos es que lo puedo hacer con cualquier punto. """
                    #todo: appendear todos los valores en la misma fila (fila==frame). Por ahora pone uno en cada fila.
                    for a in track_angle:
                        if a not in valid_body_string:
                            raise Exception(f'Body part not added to code analysis! Only can use one of {valid_body_string}')
                        id_angle = self.detector.pose_lm_dict[a]
                        self.df_results = self.df_results.append(
                            {f'{a}_angle': self.detector.findAngle(img, (id_angle - 2), id_angle, (id_angle + 2))}
                            ,ignore_index=True)

                    # add to results foot and shoulders heights log (in  lmlist[id][2] 2 is for cy) :
                    for h in track_heights:
                        if h not in valid_body_string:
                            raise Exception(f'Body part not added to code analysis! Only can use one of {valid_body_string}')
                        id_height=self.detector.pose_lm_dict[h]
                        self.df_results = self.df_results.append(
                            {f'{h}_height': self.detector.lmlist[id_height][2]},
                            ignore_index=True)

                if show_video:
                    cv2.imshow("image", img)
                cv2.waitKey(120)

                # todo: agregar tipito con palitos solo sobre video
                # detector.create_landmark_graph()

                # si apreto la "q" mientras corre, cierra el codigo:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:  # else == si frame es false --> termina el video
                return self.df_results
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

        if plot:
            self.detector.scatter_log(self.df_results)


if __name__ == "__main__":
    video = Video('corte_cajon_igna_310.mp4')
    video.run_analysis()
    df = video.df_results
    #todo: sacar VALORES nulos de df_results (dropna no sirve porque saca filas/col enteras nulas, quiero valores nomas)
    df_final = df[df.notna()]
    print(df_final)

# TODO: observaciones: mejorar el seguimiento --> puntos inestables y al botin lo pierde ene l momento del impacto!
## aumentÉ trackcon
## aumentÉ waitKEy
# TODO: mediciones, velocidad, angulos, alturas

# todo:comparar todas las patadas del mismo jugador para armar promedios --> mejorar
# todo: comparar contra jugadores profesional y sus numeros. --> conseguir video
# todo: object detection para detectar la pelota y visualizar cuando/donde (altura) impacta a la pelota.
