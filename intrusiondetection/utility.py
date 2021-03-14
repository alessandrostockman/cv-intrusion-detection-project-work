import numpy as np
import cv2
from matplotlib import pyplot as plt

from intrusiondetection.parameters import ParameterSet
from intrusiondetection.morphology import MorphOp, MorphOpsSet

# Distance Functions

def distance_manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    return (np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)).astype(np.uint8)

# Plotting Functions

def subplot_images(image_dicts):
    plt.figure(figsize=(20, 10))
    for index, image_dict in enumerate(image_dicts):
        plt.subplot(1, len(image_dicts), index+1)
        plt.axis('off')
        if image_dict['title'] is None:
             image_dict['title'] = None
        image_dict['object'].display(image_dict['key'], title=image_dict['title'], show=False)

# Parameter Utilities

def default_parameters():
    return ParameterSet({
        "input_video": "rilevamento-intrusioni-video.avi",
        "output_directory": "output",
        "output_streams": {
            #'foreground': ['image_output', 'blobs_detected', 'blobs_classified', 'image_blobs', 'blobs_remapped', 'blobs_labeled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined',],
            #'background': ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']
            'foreground': ['image_output'],
            'background': []
        }

    }, {
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
    })