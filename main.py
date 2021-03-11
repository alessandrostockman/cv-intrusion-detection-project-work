import numpy as np
import cv2

from intrusiondetection.parameters import ParameterList
from intrusiondetection.utility import distance_euclidean
from intrusiondetection.morphology import MorphOp, MorphOpsSet
from intrusiondetection.video import Video
from intrusiondetection.displayable import Background

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "output_streams": {
        'foreground': ['image_output', 'blobs_detected', 'blobs_classified', 'image_blobs', 'blobs_remapped', 'blobs_labeled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined',],
        #'foreground': ['subtraction', 'mask_raw', 'mask_0', 'mask_1', 'mask_2'],
        #'background': ['blind', 'subtraction', 'mask_raw', 'mask_0', 'mask_1', 'mask_2']
        'background': ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']
    },
    "initial_background_frames": [80],
    "initial_background_interpolation": [np.median],
    "background_alpha": [0.3],
    "background_threshold": [30],
    "background_distance": [distance_euclidean],
    "background_morph_ops": [
        MorphOpsSet(
            MorphOp(cv2.MORPH_OPEN, (3,3)), 
            MorphOp(cv2.MORPH_CLOSE, (50,50), cv2.MORPH_ELLIPSE), 
            MorphOp(cv2.MORPH_DILATE, (15,15), cv2.MORPH_ELLIPSE)
        )
    ],
    "threshold": [15],
    "distance": [distance_euclidean],
    "morph_ops": [
        MorphOpsSet(
            MorphOp(cv2.MORPH_OPEN, (3,3)),
            MorphOp(cv2.MORPH_CLOSE, (50, 50), cv2.MORPH_ELLIPSE),
            MorphOp(cv2.MORPH_OPEN, (10,10), cv2.MORPH_ELLIPSE),
        )
    ],
    "similarity_threshold": 80,
    "classification_threshold": 2.6,
    "edge_threshold": 92,
    "edge_adaptation": 0.1
})

for params in param_bag:
    initial_background = Background(
        input_video_path=params.input_video, 
        interpolation=params.initial_background_interpolation, 
        frames_n=params.initial_background_frames
    )
    
    video = Video(params.input_video)
    video.intrusion_detection(params, initial_background)
