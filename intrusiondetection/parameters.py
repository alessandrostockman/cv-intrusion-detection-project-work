import itertools
import numpy as np
import cv2

from intrusiondetection.model import Background

class ParameterList:

    def __init__(self, params):
        global_keys = {'input_video', 'output_directory', 'output_streams'}
        tuning_keys = {
            'initial_background_frames', 'initial_background_interpolation', 'background_threshold', 'background_distance', 'background_alpha', 
            'background_morph_ops', 'threshold', 'distance', 'morph_ops', 'similarity_threshold', 'classification_threshold', 'edge_threshold'
        }

        global_params = {key: val for key, val in params.items() if key in global_keys}

        tuning_params = {key: [val] if not isinstance(val, list) else val if len(val) > 0 else [None] for key, val in params.items() if key in tuning_keys}

        self.items = (ParameterSet(global_params, dict(zip(tuning_params, x))) for x in itertools.product(*tuning_params.values()))

    def __iter__(self):
        return self.items

class ParameterSet:

    def __init__(self, global_params, tuning_params):
        self.output_directory = global_params['output_directory']
        self.input_video = global_params['input_video']
        self.output_streams = global_params['output_streams']

        self.initial_background_frames = tuning_params['initial_background_frames']
        self.initial_background_interpolation = tuning_params['initial_background_interpolation']
        self.background_alpha = tuning_params['background_alpha']
        self.background_morph_ops = tuning_params['background_morph_ops']
        self.background_threshold = tuning_params['background_threshold']
        self.background_distance = tuning_params['background_distance']
        self.threshold = tuning_params['threshold']
        self.distance = tuning_params['distance']
        self.morph_ops = tuning_params['morph_ops']
        self.similarity_threshold = tuning_params['similarity_threshold']
        self.classification_threshold = tuning_params['classification_threshold']
        self.edge_threshold = tuning_params['edge_threshold']

        self.output_video = str(self) + ".avi"
        self.output_text = str(self) + ".csv"


    def __str__(self):
        '''
        Returns a string representing uniquely the current set of parameters, used for generating the output files names
        '''
        
        return "{}/tuning".format(self.output_directory)
        return "{}/tuning_{}_{}_{}_{}_{}".format(
            self.output_directory, 
            self.background,
            self.threshold, 
            self.distance.__name__,
            int(self.alpha * 100),
            self.morph_ops
        )

