import numpy as np
import cv2

from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.utility.parameters import ParameterList
from intrusiondetection.model.morphop import MorphOp
from intrusiondetection.utility.distance import euclidean

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [55],
    "distance": [euclidean],
    "alpha": [0.02],
    "morphology_operations": [
        [MorphOp(cv2.MORPH_OPEN, (5,5)), MorphOp(cv2.MORPH_CLOSE, (30, 50), cv2.MORPH_RECT)]
    ],
    "background": {
        "frames": [60],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    ida = IntrusionDetectionAlgorithm(params)
    ida.execute()