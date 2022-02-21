import video_pose as video


if __name__ == "__main__":
    video = video.Video('corte_cajon_igna_310.mp4')
    video.run_analysis(show_video=False, plot=True)
    df = video.df_results


# TODO: observaciones: mejorar el seguimiento --> puntos inestables y al botin lo pierde ene l momento del impacto!
# TODO: mediciones, velocidad, angulos, alturas
# TODO: comparar todas las patadas del mismo jugador para armar promedios --> mejorar
# TODO: comparar contra jugadores profesional y sus numeros. --> conseguir video
# TODO: object detection para detectar la pelota y visualizar cuando/donde (altura) impacta a la pelota.
