import numpy as np
import cv2

from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.utility.parameters import ParameterList
from intrusiondetection.model.morphop import MorphOp
from intrusiondetection.model.morphop import MorphOpsSet
from intrusiondetection.utility.distance import *

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [37],
    "distance": [euclidean],
    "background_morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_CLOSE, (3,3), iterations=1),MorphOp(cv2.MORPH_OPEN, (3,3), iterations=2), MorphOp(cv2.MORPH_DILATE, (25,10)), MorphOp(cv2.MORPH_ERODE, (15,5)))
    ],
    "alpha": [0.1],
    "morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (3,3)), MorphOp(cv2.MORPH_CLOSE, (5,5), iterations=3), MorphOp(cv2.MORPH_CLOSE, (40,3)))#MorphOp(cv2.MORPH_CLOSE, (10,3), iterations=3))
    ],
    "background": {
        "frames": [100],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    ida = IntrusionDetectionAlgorithm(params)
    ida.execute()