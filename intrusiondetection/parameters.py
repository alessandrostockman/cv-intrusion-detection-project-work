import itertools
import numpy as np
import cv2

from intrusiondetection.model import Background

class ParameterList:

    def __init__(self, params):
        global_keys = {'input_video', 'output_directory'}
        tuning_keys = {'threshold', 'distance', 'background_threshold', 'background_distance', 'alpha', 'background', 'morph_ops', 'background_morph_ops'}

        global_params = {key: val for key, val in params.items() if key in global_keys}

        background = params['background']
        params['background'] = self.setup_backgrounds(global_params['input_video'], background)

        tuning_params = {key: [val] if not isinstance(val, list) else val if len(val) > 0 else [None] for key, val in params.items() if key in tuning_keys}

        self.items = (ParameterSet(global_params, dict(zip(tuning_params, x))) for x in itertools.product(*tuning_params.values()))

    def __iter__(self):
        return self.items

    def setup_backgrounds(self, input_video_path, background):
        ''' 
            Initializes the parameters set for the background subtraction phase,
            Returns a list of sets containing the parameters of that run.
        '''
        bs = []
        for params in (dict(zip(background, x)) for x in itertools.product(*background.values())):
            bs.append(Background(input_video_path=input_video_path, interpolation=params["interpolation"], frames_n=params["frames"]))
        return bs

class ParameterSet:

    def __init__(self, global_params, tuning_params):
        self.output_directory = global_params['output_directory']
        self.input_video = global_params['input_video']

        self.threshold = tuning_params['threshold']
        self.distance = tuning_params['distance']
        self.alpha = tuning_params['alpha']
        self.background = tuning_params['background']
        self.morph_ops = tuning_params['morph_ops']
        self.background_morph_ops = tuning_params['background_morph_ops']
        self.background_threshold = tuning_params['background_threshold']
        self.background_distance = tuning_params['background_distance']

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

