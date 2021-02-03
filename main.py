from intrusiondetection.algorithm import IntrusionDetectionAlgorithm
from intrusiondetection.utility.parameters import ParameterList
from intrusiondetection.utility.distance import euclidean
import numpy as np

param_bag = ParameterList({
    "input_video": "rilevamento-intrusioni-video.avi",
    "output_directory": "output",
    "threshold": [55],
    "distance": [euclidean],
    "alpha": [0.02, 0.05],
    # "morphology": [bm_test],
    "background": {
        "frames": [60],
        "interpolation": [np.median]
    }
})

for params in param_bag:
    ida = IntrusionDetectionAlgorithm(params)
    ida.execute()