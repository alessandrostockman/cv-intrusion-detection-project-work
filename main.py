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
        MorphOpsSet(MorphOp(cv2.MORPH_OPEN, (3,3)), MorphOp(cv2.MORPH_CLOSE, (5,5), iterations=3), MorphOp(cv2.MORPH_CLOSE, (40,3)))#MorphOp(cv2.MORPH_CLOSE, (10,3), iterations=3))
    ],
    "background_threshold": [37],
    "background_distance": [distance_euclidean],
    "background_morph_ops": [
        MorphOpsSet(MorphOp(cv2.MORPH_CLOSE, (3,3), iterations=1),MorphOp(cv2.MORPH_OPEN, (3,3), iterations=2), MorphOp(cv2.MORPH_DILATE, (25,10)), MorphOp(cv2.MORPH_ERODE, (15,5)))
    ],
    "alpha": [0.1],
    "background": {
        "frames": [100],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    video = Video("rilevamento-intrusioni-video.avi")
    video.intrusion_detection(params, params.background)
