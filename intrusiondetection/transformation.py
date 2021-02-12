import csv
import numpy as np
import cv2
<<<<<<< HEAD
from intrusiondetection.model.blob import Blob

from copy import copy

=======

from intrusiondetection.model import Blob
>>>>>>> refactor

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
        
        blobs = []
        countours_frame = np.zeros_like(self.parameters.frame)
        new_labels = 0
        check_diss = len(self.parameters.previous_blobs) > num_labels #TODO Remove
        previous_blobs = copy(self.parameters.previous_blobs)
        for label in range(1, num_labels):
            blob_image = np.zeros(labels.shape, dtype=np.uint8) 
            blob_image[labels == label] = 1
            contour = self.parse_contours(blob_image)

            blob = Blob(contour, self.parameters.frame)
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
            if blob.is_true_object(60):
                hue_color = 179 / blob.label
                color = (int(hue_color), 255, 255)
            else:
                hue_color = 90
                color = (int(hue_color), 128, 128)
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
        
        '''
        final_blob_frame = np.zeros(blob_image.shape, dtype=np.uint8)
        for blob in blobs:
            blob.test(self.parameters.frame)
            blob_frame = blob.image * blob.g
            final_blob_frame = final_blob_frame + blob_frame.astype(np.uint8)
        final_blob_frame = np.tile(final_blob_frame[:,:,np.newaxis], 3)
        '''
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
