import sys
import argparse
import time
import numpy as np
import cv2

from intrusiondetection.parameters import ParameterList
from intrusiondetection.video import Video
from intrusiondetection.displayable import Background
from intrusiondetection.presets import default_preset
from intrusiondetection.parameters import ParameterSet
from intrusiondetection.enum import ParameterPreset

def execute_intrusion_detection(input_path, output_dir, tuning_mode=False, compute_stats=False):
    if tuning_mode:
        streams = {
            'foreground': ['image_output', 'blobs_detected', 'blobs_classified', 'image_blobs', 'blobs_remapped',
                'blobs_labeled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined',],
            'background': ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']
        }
    else:
        streams = {
            'foreground': ['image_output'],
            'background': []
        }

    params = ParameterSet(global_params={
        "input_video": input_path,
        "output_directory": output_dir,
        "output_streams": streams
    }, tuning_params=default_preset(ParameterPreset.FAST))
    
    initial_background = Background(
        input_video_path=params.input_video, 
        interpolation=params.initial_background_interpolation, 
        frames_n=params.initial_background_frames
    )

    video = Video(params.input_video)
    stats_data = video.intrusion_detection(params, initial_background, tuning=tuning_mode, stats=compute_stats)

    if stats_data is not None:
        print_data = lambda title, arr : print(title, "-", "Max:", max(arr), "- Min:", min(arr), "- Avg:", round(sum(arr) / len(arr),2))
        print_data("Times", stats_data['times'])
        for idx, data in stats_data['blobs'].items():
            print("Data for BLOB #" + str(idx))
            print_data("Edge Scores", data['edge_scores'])
            print_data("Classification Scores", data['classification_scores'])

if __name__ == "__main__":
    start_time = time.time()
    execute_intrusion_detection("input.avi", "output", tuning_mode=False, compute_stats=False)
    print("Computation finished in {} seconds".format(round(time.time() - start_time, 4)))
