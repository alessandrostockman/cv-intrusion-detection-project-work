import numpy as np
import cv2

from intrusiondetection.parameters import ParameterList
from intrusiondetection.utility import distance_euclidean
from intrusiondetection.model import MorphOp, MorphOpsSet, Video

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [37],
    "distance": [distance_euclidean],
    "morph_ops": [
        MorphOpsSet(
            MorphOp(cv2.MORPH_OPEN, (3,3)),
            MorphOp(cv2.MORPH_CLOSE, (45,70), cv2.MORPH_ELLIPSE),
            MorphOp(cv2.MORPH_OPEN, (6,6), cv2.MORPH_ELLIPSE),
        )
    ],
    "background_threshold": [37],
    "background_distance": [distance_euclidean],
    "background_morph_ops": [
        MorphOpsSet(
            MorphOp(cv2.MORPH_OPEN, (3,3), iterations=1), 
            MorphOp(cv2.MORPH_CLOSE, (3,3), iterations=1), 
            MorphOp(cv2.MORPH_DILATE, (25,25), cv2.MORPH_ELLIPSE)
        )    
    ],
    "alpha": [0.1],
    "background": {
        "frames": [80],
        "interpolation": [np.median]
    },
    "similarity_threshold": 5000,
    "classification_threshold": 100,
    "edge_threshold": 2000,
})

for params in param_bag:
    video = Video("rilevamento-intrusioni-video.avi")
    backgrounds = video.process_backgrounds('selective', params.background, params.alpha, params.background_threshold, params.background_distance, params.background_morph_ops)
    video.intrusion_detection(params, backgrounds)
