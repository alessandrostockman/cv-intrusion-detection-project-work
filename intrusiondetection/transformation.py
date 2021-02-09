import csv
import numpy as np
import cv2

from intrusiondetection.model import Blob

class ChangeDetectionTransformation:

    def __init__(self, parameters):
        self.parameters = parameters

    def apply(self, frame):
        ''' 
            Calls the background subtraction method using {<image> <background image> <Distance function> <Threshold> ...<>} as inputs
            Returns a matrix of boolean corresponding to the foreground pixels
        '''
        self.update_selective_background(frame)
        mask = self.background_subtraction(frame, self.parameters.adaptive_background)
        return mask

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

class BinaryMorphologyTransformation:

    def __init__(self, parameters):
        self.parameters = parameters

    def apply(self, mask):
        '''
            Calls the the method contained in parameters['morphology'] and returns the resulting mask after applying the morphology as a matrix of Boolean TODO: Assicurarsi che la morphology returna sempre una matrice
            di booleani, se non lo fa, accertarsi di cambiare questa descrizione
        '''
        mask = mask.astype(np.uint8)
        for op in self.parameters.morph_ops.get():
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        return mask

class ConnectedComponentTransformation:
        
    def __init__(self, parameters):
        self.parameters = parameters

    def apply(self, mask):
        '''Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        _, thresh = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(thresh)

        if num_labels <= 1:
            return mask, mask, []

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        blob_frame = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        blob_frame = cv2.cvtColor(blob_frame, cv2.COLOR_HSV2BGR)
        blob_frame[label_hue==0] = 0
        
        blobs = []
        countours_frame = np.zeros_like(self.parameters.frame)
        for label in range(1, num_labels):
            hue_color = 179*label/np.max(labels)
            tmp = np.zeros(labels.shape, dtype=np.uint8)
            tmp[labels == label] = 255
            contour = self.parse_contours(tmp)
            blobs.append(Blob(label, contour))

            color = (int(hue_color), 255, 255)
            cv2.drawContours(countours_frame, contour, -1, color, 3)
            countours_frame = cv2.cvtColor(countours_frame, cv2.COLOR_HSV2BGR)

        return countours_frame, blob_frame, blobs

    def parse_contours(self, thresh):
        ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours
