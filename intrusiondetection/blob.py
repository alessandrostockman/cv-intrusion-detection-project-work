
import math
from copy import copy
import numpy as np
import cv2

from intrusiondetection.enum import BlobClass
from intrusiondetection.utility import hue_to_rgb

class Blob:

    color_palette = { 
        BlobClass.OBJECT: (255, 0, 0),
        BlobClass.PERSON: (0, 255, 0),
        BlobClass.FAKE: (0, 0, 255),
    }

    def __init__(self, label, mask, original_frame):
        self.label = label
        self.contours = self.parse_contours(mask)
        self.main_contours = self.contours[0]
        self.mask = mask
        self.frame = original_frame

        self.id = None
        self.compute_features()

        self.__cs = None
        self.__es = None
        self.blob_class = None
        self.is_present = True
        self.previous_match = None

    def __str__(self):
        name = ""
        if self.is_present:
            name = str(self.blob_class)
        else:
            name = "FAKE"
        return str(self.classification_score()) + " " + name

    def compute_features(self):
        moments = cv2.moments(self.main_contours)
        if moments['m00'] == 0: #TODO Remove
            moments['m00'] = 1
            print("ERR")

        self.perimeter = cv2.arcLength(self.main_contours, True)
        self.area = moments['m00']
        self.cx = int(moments['m10']/self.area)
        self.cy = int(moments['m01']/self.area)

    def attributes(self):
        return [self.id, self.area, self.perimeter, self.blob_class]

    def parse_contours(self, image):
        ret = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours

    def search_matching_blob(self, candidate_blobs, dissimilarity_threshold):
        '''
            Detcting matching blobs using the dissimilarity method shown below
        '''
        match = None
        if len(candidate_blobs) > 0:
            best_blob = None
            best_dissimilarity = math.inf
            best_index = -1
            for index, candidate_blob in enumerate(candidate_blobs):
                dissimilarity = self.dissimilarity_score(candidate_blob)
                if dissimilarity < dissimilarity_threshold and dissimilarity < best_dissimilarity:
                    best_blob = candidate_blob
                    best_dissimilarity = dissimilarity
                    best_index = index

            if best_blob is not None:
                match = best_blob
                candidate_blobs.pop(best_index)
        return match

    def dissimilarity_score(self, other):
        '''
            Calculating the dissimilarity of two blobs, as lower the dissimilarity as more likely the two blobs represent the same one in two different frames
        '''
        #TODO Improve
        area_diff = abs(other.area - self.area)
        barycenter_diff = abs((other.cx - self.cx) + (other.cy - self.cy))
        return area_diff + barycenter_diff

    def classification_score(self):
        if self.__cs == None:
            self.__cs = self.area
        return self.__cs

    def edge_score(self):
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

            for coord in self.main_contours:
                y, x = coord[0][0], coord[0][1]

                window = self.frame[x-1:x+2,y-1:y+2]
                if window.shape == (3, 3):
                    val += np.maximum(abs((window * mat_x).sum()), abs((mat_y * window).sum()))
                
            self.__es = val / len(self.main_contours)

            if self.previous_match is not None:
                #TODO parametrize
                curr_weight = 0.2
                new = self.__es * curr_weight + self.previous_match.edge_score() * (1 - curr_weight)
        return self.__es

    def classify(self, classification_threshold):
        #Distinguish wether a blob is a person or an object in base of the are of his blob 
        self.blob_class = BlobClass.PERSON if self.classification_score() > classification_threshold else BlobClass.OBJECT
        self.color = self.color_palette[self.blob_class]
        return self.blob_class

    def detect(self, edge_threshold):
        #detecting true blob from fake one in base of the gradient of the value of the edge
        self.is_present = self.edge_score() > edge_threshold
        if not self.is_present:
            self.color = self.color_palette[BlobClass.FAKE]
        return self.is_present

    def write_text(self, image, text, scale=.5, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, scale, thickness)[0]
        offset = textsize[0] // 2
        #center = (self.cx - offset, self.cy - offset)
        center = (self.cx, self.cy)
        cv2.putText(image, text, center, font, scale, (0,255,0), thickness, cv2.LINE_AA)

