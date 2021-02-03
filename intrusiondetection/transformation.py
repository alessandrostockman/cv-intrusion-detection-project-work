from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np
import csv

from intrusiondetection.utility.frame import binarize_mask
from intrusiondetection.model.blob import Blob

class Transformation(ABC):

    @abstractmethod
    def __init__(self, params):
        pass

    @abstractmethod
    def apply(self, frame):
        pass

class ChangeDetectionTransformation(Transformation):

    def __init__(self, parameters):
        self.parameters = parameters

        self.initialize_subtractor()

    def apply(self, frame):
        ''' 
            Calls the background subtraction method using {<image> <background image> <Distance function> <Threshold> ...<>} as inputs
            Returns a matrix of boolean corresponding to the foreground pixels
        '''
        #mask = self.background_subtraction_mog2(frame)
        #mask = self.background_subtraction(frame, TODO: background)
        background = self.blind_background(frame, self.parameters.alpha)
        self.parameters.adaptive_background = background
        mask = self.background_subtraction(frame, background)
        return mask

    def two_frame_difference(self, prev_frame, curr_frame):
        ''' 
            Applies the distance function on the two frames returning a boolean matrix
        '''
        return self.parameters.distance(curr_frame, prev_frame) > self.parameters.threshold

    def three_frame_difference(self, prev_frame, curr_frame, next_frame):
        '''
            Computes the TFD between the first and second frame and the TFD between the second and the third frame, after that it computes and returns as matrix of Booleans the logical and between the two TFD.
        '''
        diff1 = two_frame_difference(prev_frame, curr_frame)
        diff2 = two_frame_difference(curr_frame, next_frame)
        and_mask = np.prod([diff1, diff2],axis=0, dtype=bool)
        return and_mask

    def background_subtraction(self, frame, background):
        '''
            Computes the Background subtraction (distance(frame,background)) and returns a matrix of boolean
        '''
        frame = frame.astype(float)
        mask = self.parameters.distance(frame, background) > self.parameters.threshold
        return mask


    def blind_background(self, frame, alpha):
        new_bg = self.parameters.adaptive_background * (1-alpha)
        new_bg = new_bg + frame * alpha
        
        return new_bg
        
    def background_set_initialization(self, input_video_path, parameter_set):
        ''' 
            TODO: Initializes the parameters set for the background subtraction phase,
            Returns a list of sets containing the parameters of that run.
        '''
        bs = []
        for params in compute_parameters(parameter_set):
            cap = cv2.VideoCapture(input_video_path)
            bs.append({
                "image": background_initialization(cap, params["interpolation"], params["frames"]),
                "name": "{}_{}".format(params["frames"], params["interpolation"].__name__)
            })
        return bs
        

    def background_initialization(self, cap, interpolation, n=100):
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

    def initialize_subtractor(self):
        '''
            Creates a MOG2 background subtractor
        '''
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgbg.setDetectShadows(False)

    def background_subtraction_mog2(self, frame):
        '''
            Applies the MOG2 background subtractor to the frame and returns the foreground mask as 8-bit binary image
        '''
        return self.fgbg.apply(frame)

class BinaryMorphologyTransformation(Transformation):

    def __init__(self, parameters):
        self.parameters = parameters

    def apply(self, mask):
        '''
            Calls the the method contained in parameters['morphology'] and returns the resulting mask after applying the morphology as a matrix of Boolean TODO: Assicurarsi che la morphology returna sempre una matrice
            di booleani, se non lo fa, accertarsi di cambiare questa descrizione
        '''
        return self.bm_test(mask)
        if self.parameters.morphology is None:
            return mask
        return self.parameters.morphology(mask)

    def bm_test(self, mask):
        '''
            Applies the binary morphology on the mask and returns it as a matrix of Boolean
        '''
        mask = mask.astype(np.uint8)
        for op in self.parameters.morphology_operations:
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel)
        return mask

class ConnectedComponentTransformation(Transformation):
        
    def __init__(self, parameters):
        self.parameters = parameters
        self.initialize_blob_detector()

    def apply(self, mask):
        '''Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        ret, thresh = cv2.threshold(mask.astype(np.uint8), 0, 255, 0)
        ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret
        colored_frame = self.parameters.frame.copy()
        cv2.drawContours(colored_frame, contours, -1, (0,255,0), 3)
        
        blobs = []
        for contour in contours:
            blobs.append(Blob(1))
        return colored_frame, blobs
    
    def initialize_blob_detector(self):
        '''Initializes the parameters of the blob detector and returns the detector
        '''
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        #params.minThreshold = 10
        #params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = False
        #params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = False
        #params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        #params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        #params.minInertiaRatio = 0.01
        
        params.filterByColor = True
        params.blobColor = 255
        
        self.detector = cv2.SimpleBlobDetector_create(params)

    def blob_detector(self, mask):
        '''
            Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        keypoints = self.detector.detect(mask)
        #frame_parsed = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoints

    def append_text_output(frame_index, blobs, csv_writer):
        '''Writes on the csv document the infos of the blob.
        '''
        csv_writer.writerow([frame_index, len(blobs)])
        for blob in blobs:
            csv_writer.writerow([blob['label'], blob['area'], blob['perimeter'], blob['classification']])