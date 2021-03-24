import math
from copy import copy
import numpy as np
import cv2

from intrusiondetection.enum import BlobClass

class Blob:

    color_map = { 
        BlobClass.OBJECT: 6, #(255, 0, 0),
        BlobClass.PERSON: 7, #(0, 255, 0),
        BlobClass.FAKE: 8, #(0, 0, 255),
    }

    color_palette = [
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    ]

    def __init__(self, label, mask, original_frame):
        self.id = None
        self.previous_match = None
        self.label = label
        self.contours = self.parse_contours(mask)
        self.mask = mask

        self.__cs = None
        self.__es = None
        self.__main_contours = self.contours[0]
        self.__frame = original_frame
        self.__frame_height, self.__frame_width = self.mask.shape
        self.__frame_pixels = self.__frame_width * self.__frame_height
        self.__frame_diag = math.sqrt(self.__frame_width ** 2 + self.__frame_height ** 2)
        self.__blob_class = None
        self.__is_present = None
        
        self.compute_features()

    def compute_features(self):
        '''
            Wrapper method for the computation of all the blob features used
        '''
        moments = cv2.moments(self.__main_contours)

        self.perimeter = round(cv2.arcLength(self.__main_contours, True), 2)
        self.area = round(moments['m00'], 2)
        self.cx = int(moments['m10'] / self.area)
        self.cy = int(moments['m01'] / self.area)

    def attributes(self):
        '''
            List of attributes printed in the text output
        '''
        return [self.id, self.area, self.perimeter, self.cx, self.cy, self.__cs, self.__es, self.__is_present, self.__blob_class]

    def parse_contours(self, image):
        '''
            Returns the contours of the blob found by OpenCV's findContours function
        '''
        ret = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours

    def search_matching_blob(self, candidate_blobs, similarity_threshold):
        '''
            Searches for a correspondence for the current blob in a list of candidates based on their similarity score
        '''
        match = None
        if len(candidate_blobs) > 0:
            best_blob = None
            best_similarity = 0
            best_index = -1
            for index, candidate_blob in enumerate(candidate_blobs):
                similarity = self.similarity_score(candidate_blob)
                if similarity > similarity_threshold and similarity > best_similarity:
                    best_blob = candidate_blob
                    best_similarity = similarity
                    best_index = index

            if best_blob is not None:
                match = best_blob
                candidate_blobs.pop(best_index)
        return match

    def similarity_score(self, other):
        '''
            Computation of two blobs' normalized similarity score
        '''
        area_diff = abs(other.area - self.area)
        area_diff_norm = area_diff / self.__frame_pixels
        barycenter_dist = math.sqrt((other.cx - self.cx) ** 2 + (other.cy - self.cy) ** 2)
        barycenter_dist_norm = barycenter_dist / self.__frame_diag
        return round((1 - ((area_diff_norm + barycenter_dist_norm) / 2)) * 100, 2)

    def classification_score(self):
        '''
            Lazy computation of the classification score performed by normalizing the blob area
        '''
        if self.__cs == None:
            self.__cs = round(self.area / self.__frame_pixels * 100, 2)
        return self.__cs

    def edge_score(self, edge_adaptation=1):
        '''
            Lazy computation of the edge score performed by computing the Sobel operator over all the blob countours pixels and weights 
            it with respect to the edge score of its match in the previous frame if present
        '''
        #Calculating the derivatives to obtain the gradient value to verify if the object is a true object or a fake one
        if self.__es == None:
            val = 0
            mat_x = np.flip(np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ]))
            mat_y = np.flip(np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]))

            counter_pixels = 0
            for coord in self.__main_contours:
                y, x = coord[0][0], coord[0][1]

                window = self.__frame[x-1:x+2,y-1:y+2]
                if window.shape == (3, 3):
                    counter_pixels += 1
                    val += np.maximum(abs((window * mat_x).sum()), abs((mat_y * window).sum()))
                
            self.__es = val / counter_pixels
            
            if self.previous_match is not None:
                self.__es = self.__es * edge_adaptation + self.previous_match.edge_score() * (1 - edge_adaptation)
            
            self.__es = round(self.__es, 2)
        return self.__es

    def classify(self, classification_threshold):
        '''
            Classifies the blob into PERSON or OBJECT using the classification score and sets the blob color for the final output
        '''
        #Distinguish wether a blob is a person or an object in base of the are of his blob 
        self.__blob_class = BlobClass.PERSON if self.classification_score() > classification_threshold else BlobClass.OBJECT
        self.color = self.color_palette[self.color_map[self.__blob_class]]
        return self.__blob_class

    def detect(self, edge_threshold, edge_adaptation):
        '''
            Determines the presence of the blob by thresholding the edge score and sets the blob color for the final output
        '''
        self.__is_present = self.edge_score(edge_adaptation=edge_adaptation) > edge_threshold
        if not self.__is_present:
            self.color = self.color_palette[self.color_map[BlobClass.FAKE]]
        return self.__is_present

    def write_text(self, image, text, scale=.4, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(text)
        x_offset, y_offset = cv2.getTextSize(text, font, scale, thickness)[0]
        x_offset, y_offset =  x_offset , y_offset 
        center = (self.cx - x_offset// 2, self.cy + y_offset// 2)
        cv2.putText(image, text, center, font, scale, (255,255,255), thickness, cv2.LINE_AA)

