import numpy as np
import cv2

from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.parameters import ParameterList
from intrusiondetection.utility import distance_euclidean
from intrusiondetection.model import MorphOp, MorphOpsSet, Video

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [65],
    "distance": [distance_euclidean],
    "background_morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (3,3)), MorphOp(cv2.MORPH_CLOSE, (5,8)))
    ],
    "alpha": [0.5],
    "morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (4,4)), MorphOp(cv2.MORPH_CLOSE, (20,25), cv2.MORPH_RECT))
    ],
    "background": {
        "frames": [80],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    video = Video(params)
    video.intrusion_detection()

