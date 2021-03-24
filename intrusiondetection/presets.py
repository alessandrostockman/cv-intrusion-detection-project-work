import numpy as np
import cv2

from intrusiondetection.morphology import MorphOp, MorphOpsSet
from intrusiondetection.utility import distance_euclidean
from intrusiondetection.enum import ParameterPreset

def default_preset(preset):
    return {
        ParameterPreset.FAST: preset_3,
        ParameterPreset.MEDIUM: preset_2,
        ParameterPreset.SLOW: preset_1
    }.get(preset, None)

preset_1 = {
    "initial_background_frames": 80,
    "initial_background_interpolation": np.median,
    "background_alpha": 0.3,
    "background_threshold": 30,
    "background_distance": distance_euclidean,
    "background_morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3)), 
        MorphOp(cv2.MORPH_CLOSE, (50,50), cv2.MORPH_ELLIPSE), 
        MorphOp(cv2.MORPH_DILATE, (15,15), cv2.MORPH_ELLIPSE)
    ),
    "threshold": 15,
    "distance": distance_euclidean,
    "morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3)),
        MorphOp(cv2.MORPH_CLOSE, (50, 50), cv2.MORPH_ELLIPSE),
        MorphOp(cv2.MORPH_OPEN, (10,10), cv2.MORPH_ELLIPSE),
    ),
    "similarity_threshold": 80,
    "classification_threshold": 2.6,
    "edge_threshold": 92,
    "edge_adaptation": 0.1
}

preset_2 = {
    "initial_background_frames": 80,
    "initial_background_interpolation": np.median,
    "background_alpha": 0.3,
    "background_threshold": 30,
    "background_distance": distance_euclidean,
    "background_morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3)), 
        MorphOp(cv2.MORPH_CLOSE, (55,55)), 
        MorphOp(cv2.MORPH_DILATE, (15,15))
    ),
    "threshold": 15,
    "distance": distance_euclidean,
    "morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3), cv2.MORPH_ELLIPSE),
        MorphOp(cv2.MORPH_CLOSE, (50, 50), cv2.MORPH_ELLIPSE),
        MorphOp(cv2.MORPH_OPEN, (15,15)),
    ),
    "similarity_threshold": 80,
    "classification_threshold": 2.6,
    "edge_threshold": 110,
    "edge_adaptation": 0.3
}

preset_3 = {
    "initial_background_frames": 80,
    "initial_background_interpolation": np.median,
    "background_alpha": 0.3,
    "background_threshold": 30,
    "background_distance": distance_euclidean,
    "background_morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3)), 
        MorphOp(cv2.MORPH_CLOSE, (55,55)), 
        MorphOp(cv2.MORPH_DILATE, (15,15))
    ),
    "threshold": 15,
    "distance": distance_euclidean,
    "morph_ops": MorphOpsSet(
        MorphOp(cv2.MORPH_OPEN, (3,3), cv2.MORPH_ELLIPSE),
        MorphOp(cv2.MORPH_CLOSE, (50, 50)),
        MorphOp(cv2.MORPH_OPEN, (15,15)),
    ),
    "similarity_threshold": 80,
    "classification_threshold": 2.6,
    "edge_threshold": 70,
    "edge_adaptation": 0.3
}