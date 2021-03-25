import itertools
import numpy as np
import cv2

class ParameterList:

    def __init__(self, params):
        '''
            Creates a ParameterSet for every combination of given parameters to assist during tuning
        '''

        global_keys = {'input_video', 'output_directory', 'output_streams'}
        tuning_keys = {
            'initial_background_frames', 'initial_background_interpolation', 'background_threshold', 'background_distance', 'background_alpha', 
            'background_morph_ops', 'threshold', 'distance', 'morph_ops', 'similarity_threshold', 'classification_threshold', 'edge_threshold', 'edge_adaptation'
        }

        global_params = {key: val for key, val in params.items() if key in global_keys}

        tuning_params = {key: [val] if not isinstance(val, list) else val if len(val) > 0 else [None] for key, val in params.items() if key in tuning_keys}

        self.__items = (ParameterSet(global_params, dict(zip(tuning_params, x))) for x in itertools.product(*tuning_params.values()))

    def __iter__(self):
        '''
            Iterates through the generated ParameterSet objects
        '''
        return self.__items

class ParameterSet:

    def __init__(self, global_params={}, tuning_params={}):
        '''
            Decodes the global and tuning parameters dictionaries
        '''
        self.output_directory = global_params.get('output_directory', "output")
        self.input_video = global_params.get('input_video', "input.avi")
        self.output_streams = global_params.get('output_streams', {
            'foreground': ['image_output', 'blobs_detected', 'blobs_classified', 'image_blobs', 'blobs_remapped',
                'blobs_labeled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined',],
            'background': ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']
        })
        self.store_outputs = global_params.get('store_outputs', False)

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
        self.edge_adaptation = tuning_params['edge_adaptation']

        self.output_base_name = str(self)


    def __str__(self):
        '''
            Returns a string representing uniquely the current set of parameters, used for generating the output files names for tuning purposes
        '''
        return "{}/tuning_{}_{}_{}_{}_{}_{}_{}".format(
            self.output_directory, 
            int(self.background_alpha * 100),
            self.background_morph_ops,
            self.background_threshold,
            self.background_distance.__name__,
            self.threshold, 
            self.distance.__name__,
            self.morph_ops
        )

