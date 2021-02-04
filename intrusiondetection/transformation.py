import csv
from abc import ABC, abstractmethod

import numpy as np

import cv2
from intrusiondetection.utility.frame import binarize_mask
from intrusiondetection.model.blob import Blob

from copy import copy


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
        self.update_selective_background(frame)
        mask = self.background_subtraction(frame, self.parameters.adaptive_background)
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
        and_mask = np.prod([diff1, diff2], axis=0, dtype=bool)
        return and_mask

    def background_subtraction(self, frame, background):
        '''
            Computes the Background subtraction (distance(frame,background)) and returns a matrix of boolean
        '''
        frame = frame.astype(float)
        mask = self.parameters.distance(frame, background) > self.parameters.threshold
        return mask


    def update_blind_background(self, frame):
        self.parameters.adaptive_background = self.compute_blind_background(frame)

    def compute_blind_background(self, frame):
        new_bg = self.parameters.adaptive_background * (1-self.parameters.alpha)
        new_bg = new_bg + frame * self.parameters.alpha
        return new_bg

   
    def binary_morph(self, mask):
        mask = mask.astype(np.uint8)
        for op in self.parameters.background_morph_ops.get():
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        return mask
    
    def update_selective_background(self, frame):
        new_bg = np.copy(self.parameters.adaptive_background)
        subtraction = self.background_subtraction(frame, self.parameters.adaptive_background)
        binary_res = self.binary_morph(subtraction)
        self.parameters.sub = np.zeros(subtraction.shape, dtype=np.uint8)
        self.parameters.mor = binary_res
        self.parameters.sub[subtraction] = 255
        self.parameters.mor[self.parameters.mor == 1] = 255

        mask1 = np.zeros(binary_res.shape, dtype=int)
        mask2 = mask1.copy()
        mask1[binary_res == 0] = 1
        mask2[binary_res != 0] = 1
        new_bg = self.compute_blind_background(frame)[:,:,0] * mask1 + new_bg[:,:,0] * mask2
        self.parameters.adaptive_background = np.tile(new_bg[:,:,np.newaxis], 3)
        
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
        for op in self.parameters.morph_ops.get():
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        return mask

class ConnectedComponentTransformation(Transformation):
        
    def __init__(self, parameters):
        self.parameters = parameters
        self.initialize_blob_detector()

    def apply(self, mask):
        '''Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        _, thresh = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(thresh)

        if num_labels <= 1:
            return mask, mask, []
        
        blobs = []
        countours_frame = np.zeros_like(self.parameters.frame)
        new_labels = 0
        check_diss = len(self.parameters.previous_blobs) > num_labels #TODO Remove
        previous_blobs = copy(self.parameters.previous_blobs)
        for label in range(1, num_labels):
            blob_image = np.zeros(labels.shape, dtype=np.uint8) 
            blob_image[labels == label] = 1
            contour = self.parse_contours(blob_image)

            blob = Blob(contour)
            label_id = None
            prev_labels_number = max([x.label for x in self.parameters.previous_blobs]) if len(self.parameters.previous_blobs) > 0 else 0
            if len(previous_blobs) > 0:
                best_blob = None
                best_dissimilarity = 1000000
                best_index = -1
                for index, candidate_blob in enumerate(previous_blobs):
                    dissimilarity = self.compute_dissimilarity(candidate_blob, blob)
                    if dissimilarity < best_dissimilarity and dissimilarity < 5000: # T to change
                        best_blob = candidate_blob
                        best_dissimilarity = dissimilarity
                        best_index = index
                    
                    if check_diss:
                        print(dissimilarity)
                
                if best_blob is not None:
                    label_id = best_blob.label
                    previous_blobs.pop(best_index)
            
            if label_id is None:
                new_labels += 1
                label_id = new_labels + prev_labels_number
                print("New object found:", label_id, "Prev Label:", prev_labels_number, "New:", new_labels)

            blob.label = label_id
            blob.image = blob_image
            blobs.append(blob)

        final_blob_frame = np.tile(np.zeros(blob_image.shape, dtype=np.uint8)[:,:,np.newaxis], 3)
        final_cont_frame = np.tile(np.zeros(blob_image.shape, dtype=np.uint8)[:,:,np.newaxis], 3)
        for blob in blobs:
            #hue_color = 179 * blob.id / (new_labels + prev_labels_number)
            hue_color = 179 / blob.label
            color = (int(hue_color), 255, 255)
            cv2.drawContours(countours_frame, contour, -1, color, 3)
            countours_frame = cv2.cvtColor(countours_frame, cv2.COLOR_HSV2BGR)
            final_cont_frame = final_cont_frame + countours_frame

            # Map component labels to hue val, 0-179 is the hue range in OpenCV
            label_hue = blob.image * hue_color
            blank_ch = 255*np.ones_like(label_hue)*blob.image
            blob_frame = cv2.merge([label_hue, blank_ch, blank_ch])

            # Converting cvt to BGR
            blob_frame = cv2.cvtColor(blob_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
            final_blob_frame = final_blob_frame + blob_frame
        
        self.parameters.previous_blobs = blobs

        return countours_frame, final_blob_frame, blobs

    def compute_dissimilarity(self, candidate_blob, blob):
        area_diff = abs(candidate_blob.area - blob.area)
        barycenter_diff = abs((candidate_blob.cx - blob.cx) + (candidate_blob.cy - blob.cy))
        return area_diff + barycenter_diff


    def parse_contours(self, thresh):
        ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours

    def old(self, mask):
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
            blobs.append(Blob(1, contour))
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
