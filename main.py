from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.utility.parameters import ParameterList
from intrusiondetection.model.morphology_operation import MorphologyOperation
from intrusiondetection.utility.distance import euclidean
import numpy as np
import cv2

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [55],
    "distance": [euclidean],
    "alpha": [0.02, 0.05],
    "morphology_operations": [
        [MorphologyOperation(cv2.MORPH_OPEN, (5,5)), MorphologyOperation(cv2.MORPH_CLOSE, (30, 50), cv2.MORPH_RECT)]
    ],
    "background": {
        "frames": [60],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    ida = IntrusionDetectionAlgorithm(params)
    ida.execute()