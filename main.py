import numpy as np
import cv2

from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.utility.parameters import ParameterList
from intrusiondetection.model.morphop import MorphOp
from intrusiondetection.model.morphop import MorphOpsSet
from intrusiondetection.utility.distance import euclidean

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [45],
    "distance": [euclidean],
    "background_morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (3,3)), MorphOp(cv2.MORPH_CLOSE, (5,8)))
    ],
    "alpha": [0.1],
    "morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (4,4)), MorphOp(cv2.MORPH_CLOSE, (10,45), cv2.MORPH_RECT, iterations=3))
    ],
    "background": {
        "frames": [80],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    ida = IntrusionDetectionAlgorithm(params)
    ida.execute()