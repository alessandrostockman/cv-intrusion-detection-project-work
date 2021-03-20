import time
import numpy as np
import cv2

from intrusiondetection.parameters import ParameterList
from intrusiondetection.video import Video
from intrusiondetection.displayable import Background
from intrusiondetection.utility import default_parameters

def execute_intrusion_detection(compute_stats=False):
    params = default_parameters()
    initial_background = Background(
        input_video_path=params.input_video, 
        interpolation=params.initial_background_interpolation, 
        frames_n=params.initial_background_frames
    )

    video = Video(params.input_video)
    stats_data = video.intrusion_detection(params, initial_background, tuning=False, stats=compute_stats)

    if stats_data is not None:
        print_data = lambda title, arr : print(title, "-", "Max:", max(arr), "- Min:", min(arr), "- Avg:", round(sum(arr) / len(arr),2))
        print_data("Times", stats_data['times'])
        for idx, data in stats_data['blobs'].items():
            print("Data for BLOB #" + str(idx))
            print_data("Edge Scores", data['edge_scores'])
            print_data("Classification Scores", data['classification_scores'])

if __name__ == "__main__":
    start_time = time.time()
    execute_intrusion_detection(compute_stats=True)
    print("Computation finished in {} seconds".format(round(time.time() - start_time, 4)))
