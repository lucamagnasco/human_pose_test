import video_pose as video


if __name__ == "__main__":
    video = video.Video('inputs/corte_cajon_igna_310.mp4')
    video.run_analysis(show_video=True, export=True)
    video.plot_df_results(export=True)
    df = video.df_results
    print(f'Done! Results DataFrame created: \n {df.head}')

# TODO: improve tracking accuracy --> unstable points. For ex: looses cleate whtn kick impact
# TODO: other measurements
# TODO: Average multiple measurements for same player and compare to others.
# todo: Add profesional player video to measure and compare to amateur for improvement with detected KPIs.
# TODO: object detection for ball. Useful for video edit and for kick analysis (ex: speed/height at impact).
