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

def execute_intrusion_detection(input_path, output_dir, preset, tuning_mode=False, compute_stats=False):
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
        "output_streams": streams,
        "store_outputs": True
    }, tuning_params=default_preset(preset))
    
    initial_background = Background(
        input_video_path=params.input_video, 
        interpolation=params.initial_background_interpolation, 
        frames_n=params.initial_background_frames
    )

    video = Video(params.input_video)
    stats_data = video.intrusion_detection(params, initial_background, tuning=tuning_mode, stats=compute_stats)

    if stats_data is not None:
        print_data = lambda title, arr : print(title, "-", "Max:", max(arr), "- Min:", min(arr), "- Avg:", round(sum(arr) / len(arr),2))
        print_data("-- Times", stats_data['times'])
        for idx, data in stats_data['blobs'].items():
            print("-- Data for BLOB #" + str(idx))
            print_data("---- Edge Scores", data['edge_scores'])
            print_data("---- Classification Scores", data['classification_scores'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', help='Input video used to compute the intrusion detection algorithm', default='input.avi')
    parser.add_argument('-O', '--output', help='Output directory where the requested output are stored', default='output')
    parser.add_argument('-S', '--stats', help='Compute and print additional info on the elaborated data', action='store_true')
    parser.add_argument('-T', '--tuning', help='Activates tuning mode, in which all the algorithm steps are generated as output videos', action='store_true')
    parser.add_argument('-P', '--preset', help='Preset of parameters used from most accurate (1) to fastest (3)', choices=['1','2','3'], default='1')

    args = parser.parse_args()

    start_time = time.time()
    execute_intrusion_detection(args.input, args.output, preset=ParameterPreset(int(args.preset)), tuning_mode=args.tuning, compute_stats=args.stats)
    print("Computation finished in {} seconds".format(round(time.time() - start_time, 4)))
