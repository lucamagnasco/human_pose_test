import cv2
import time
import pose_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(video):
    # Create video capture object:
    cap = cv2.VideoCapture(video)  # Para camara poner numero de dispositivo (compu def = 0)
    pTime = 0

    # Create detector instance:
    detector = pm.poseDetector()

    # while True:
    while cap.isOpened():
        """ isOpened() method returns a boolean that indicates whether or not the video stream is valid"""
        frame, img = cap.read()
        """.read() method returns a tuple, where the first element is a boolean and the next element is the actual video frame. 
             When the first element is True, it indicates the video stream contains a frame to read."""
        if frame == True:
            img = detector.findPose(img)

            # armo lista de listas con la posicion de cada pose_landmarks:
            lmlist = detector.findPosition(img)
            # list tiene Id para dif los landmarks --> si quiero el 14 traigo lmlist[14]

            # calcula frames by second (fps)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime  # ptime = 0 antes del while
            #imprime fps en imagen:
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)  #(255, 0, 0) es AZUL pq cv usa BGR

            if len(lmlist)!=0:
                """lo bueno de hacerlo con objetos es que lo puedo hacer con cualquier punto"""
                #right knee
                detector.log_angle.append(detector.findeAngle(img,24,26,28))
                #print(detector.log_angle)
                #right arm
                # detector.findeAngle(img, 12, 14, 16)

                #todo: appendear el angulo al df

            cv2.imshow("image", img)
            cv2.waitKey(150)

            # si apreto la "q" mientras corre, cierra el codigo:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            return pd.DataFrame(detector.log_angle, columns=['angle']).reset_index()

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    df = main('pose_videos/corte_cajon_igna_310.mp4')
    sns.scatterplot(data=df, x='index', y='angle')
    plt.show()

#TODO: observaciones: mejorar el seguimiento --> puntos inestables y al botin lo pierde ene l momento del impacto!
        ## aumentÉ trackcon
        ## aumentÉ waitKEy
#TODO: mediciones
# velocidad
# angulos
# alturas
        # Puedo ubicar pelota para ver donde la impacta? (a que altura con resp de su cuerpo, deberia pegarle en su cintura)
#todo:comparar todas las patadas del mismo jugador para armar promedios --> mejorar
#todo: comparar contra jugadores profesional y sus numeros. --> conseguir video