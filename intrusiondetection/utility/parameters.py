import numpy as np
import cv2
import itertools

def background_set_initialization(input_video_path, background):
    ''' 
        TODO: Initializes the parameters set for the background subtraction phase,
        Returns a list of sets containing the parameters of that run.
    '''
    bs = []
    for params in (dict(zip(background, x)) for x in itertools.product(*background.values())):
        cap = cv2.VideoCapture(input_video_path)
        bs.append({
            "image": background_initialization(cap, params["interpolation"], params["frames"]),
            "name": "{}_{}".format(params["frames"], params["interpolation"].__name__)
        })
    return bs
    

def background_initialization(cap, interpolation, n=100):
    '''
        Estimates the background of the given video capture by using the interpolation function and n frames
        Returning a matrix of float64
    '''
    # Loading Video
    bg = []
    idx = 0
    # Initialize the background image
    while(cap.isOpened() and idx < n):
        ret, frame = cap.read()
        if ret and not frame is None:
            frame = frame.astype(float)
            # Getting all first n images
            bg.append(frame)
            idx += 1
        else:
            break
    cap.release()

    bg_interpolated = np.stack(bg, axis=0)
    return interpolation(bg_interpolated, axis=0).astype(int)

class ParameterList:

    def __init__(self, params):
        global_keys = {'input_video', 'output_directory'}
        tuning_keys = {'threshold', 'distance', 'alpha', 'background', 'morph_ops'}

        global_params = {key: val for key, val in params.items() if key in global_keys}

        background = params['background']
        params['background'] = background_set_initialization(global_params['input_video'], background)  #TODO

        tuning_params = {key: [val] if not isinstance(val, list) else val if len(val) > 0 else [None] for key, val in params.items() if key in tuning_keys}

        self.items = (ParameterSet(global_params, dict(zip(tuning_params, x))) for x in itertools.product(*tuning_params.values()))

    def __iter__(self):
        return self.items

class ParameterSet:

    def __init__(self, global_params, tuning_params):
        self.output_directory = global_params['output_directory']
        self.input_video = global_params['input_video']

        self.threshold = tuning_params['threshold']
        self.distance = tuning_params['distance']
        self.alpha = tuning_params['alpha']
        self.background = tuning_params['background']
        self.morph_ops = tuning_params['morph_ops']

        if self.background is not None:
            self.adaptive_background = self.background['image']

        self.output_video = str(self) + ".avi"
        self.output_text = str(self) + ".csv"


    def __str__(self):
        '''
        Returns a string representing uniquely the current set of parameters, used for generating the output files names
        '''
                
        background_name = "none"
        if self.background is not None:
            background_name = self.background["name"]
        
        return "{}/tuning_{}_{}_{}_{}_{}".format(
            self.output_directory, 
            background_name,
            self.threshold, 
            self.distance.__name__,
            int(self.alpha * 100),
            self.morph_ops
        )

