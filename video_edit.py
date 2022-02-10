from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# armo lista del resto de los cajones y los corto en un loop
cajones_arranque = {'igna':[191, 310, 334, 343, 737,758, 797],
                    'otro1':[63,558, 585,620, 637, 652],
                    'otro2':[412,479,490,864, 878, 897]}
for jugador, tiempos in cajones_arranque.items():
        for t in tiempos:
            ffmpeg_extract_subclip("pose_videos/190124 JP DIXP Medio Scrum Cajon.mp4",
                           t, t+5,
                           targetname=f"pose_videos/corte_cajon_{jugador}_{t}.mp4")
# # cajones (t = 5seg)

## designacion si es buena o mala
# 1. 5:10 (igna)  - buena
# 3. 3:11 (igna)  -
# 4. 5:34 (igna) - medio. falta atravesar (levantar maas pie)
# 5. 5:43 (igna) - un poco mejor
# 15. 12:38 (igna) la hace girar para adelante? no atraviesa el pie
# 16. 13:17 (igna) estira el pie aunque pierde un poco la posicion
# 14. 12:17 (igna) estirar el tobillo para pegarle

# 9. 9:18 (otro1) cabeza y pecho encogido
# 10. 9:45 (otro1) mejor posicion upper body. levanta cabeza/pecho
# 11. 10:20 (otro1) levanta mejor pierna
# 12. 10:37 (otro1) no tanbuena
# 13. 10:52 (otro1) no impacta con el pie de lleno
# 2. 1:03 (otro1)  - medio

# 6. 6:52 (otro2) - posicion de arranque dist.
# 7. 7:59 (otro2) le pega con tobillo
# 8. 8:10 (otro2) mejor
# 17. 14:24 (otro2) buena
# 18. 14:38 otro2  le pega con el tobillo y no le gusta como sale. tiene que estirar mas el brazo para que le quede mas lejos la pelota
# 19. 14:57 otro 2 le pega con el tobillo pero es mejor la salida corporal. paso muy grande post pegada.bajar pie al mismo lugar