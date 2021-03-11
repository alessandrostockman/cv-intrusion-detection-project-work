
from copy import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt

from intrusiondetection.blob import Blob
from intrusiondetection.utility import hue_to_rgb

class Displayable:

    def display_row(self, image_dicts):
        plt.figure(figsize=(20, 10))

        for index, image_dict in enumerate(image_dicts):
            plt.subplot(1, len(image_dicts), index+1)
            img = getattr(self, image_dict['key'])
            plt.axis('off')
            if image_dict['title'] is not None:
                plt.title(image_dict['title'])

            if img.shape[-1] == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)


    def display(self, keys, title=None, show=True):
        '''if isinstance(keys, list):
            if show:
                plt.figure(figsize=(20, 10))

            for index, key in enumerate(keys):
                plt.subplot(1, len(keys), index+1)
                img = getattr(self, key)
                plt.axis('off')
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:'''
        if show:
            plt.figure(figsize=(6.4, 4.8))

        img = getattr(self, keys)
        plt.axis('off')
        if title is not None:
            plt.title(title)

        if img.shape[-1] == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if show:
            plt.show()

    def copy(self):
        return copy(self)


class Frame(Displayable):
    '''
        Class Frame containing the methods that needs to be applied on the image frame

    '''
    def __init__(self, image):
        self.image = image[:,:,0]
        self.image_triple_channel = image
        self.image_blobs = self.image_triple_channel.copy()
        self.image_output = self.image_triple_channel.copy()
        
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_labeled = None
        self.blobs_remapped = None
        self.blobs_classified = None
        self.blobs_detected = None

        self.blobs = []

    def intrusion_detection(self, params, bg, previous_blobs):
        self.apply_change_detection(bg, params.threshold, params.distance)
        self.apply_morphology_operators(params.morph_ops)
        self.apply_blob_analysis(previous_blobs, params.similarity_threshold, params.classification_threshold, params.edge_threshold, params.edge_adaptation)
        #self.apply_blob_labeling()
        #self.apply_blob_remapping(previous_blobs, params.similarity_threshold)
        #self.apply_classification(params.classification_threshold)
        #self.apply_object_recognition(params.edge_threshold)
        #self.generate_output()
    
    def apply_change_detection(self, background, threshold, distance):
        '''
            Computes the difference between the frame and the background (computed via Selective update) and computes a binary mask of type np.uint8 
        '''
        self.subtraction = background.subtract_frame(self.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)

    def apply_morphology_operators(self, morph_ops):
        '''
            Computes the binary morphology on the raw mas and assigns it on the mask_refined of the class
        '''
        self.mask_refined = morph_ops.apply(self.mask_raw)

    def apply_blob_analysis(self, previous_blobs, similarity_threshold, classification_threshold, edge_threshold, edge_adaptation):
        self.apply_blob_labeling()
        self.apply_blob_remapping(previous_blobs, similarity_threshold)
        self.apply_classification(classification_threshold)
        self.apply_object_recognition(edge_threshold, edge_adaptation)
        self.generate_output()

    def apply_blob_labeling(self, colored_output=False):
        num_labels, labels = cv2.connectedComponents(self.mask_refined)
        self.blobs_labeled = self.image_blobs.copy()

        #No blobs detected
        if num_labels > 1:
            for curr_label in range(1, num_labels):
                blob_mask = np.where(labels == curr_label, 255, 0).astype(np.uint8)
                blob = Blob(curr_label, blob_mask, self.image)
                blob.label = curr_label
                self.blobs.append(blob)

                self.image_blobs[blob_mask > 0] = hue_to_rgb(curr_label * 179 / (num_labels - 1))
                blob.write_text(self.blobs_labeled, str(blob.label))

    def apply_blob_remapping(self, previous_blobs, similarity_threshold):
        candidate_blobs = copy(previous_blobs)
        self.blobs_remapped = self.image_blobs.copy()
        
        new_ids = 0
        for blob in self.blobs:
            matched_blob = blob.search_matching_blob(candidate_blobs, similarity_threshold)

            #if we canno't match the blob to a previous blob then it means that we need to associate a new label on the detected blob
            if matched_blob is None:
                prev_ids_number = max([x.id for x in previous_blobs], default=0)
                new_ids += 1
                matched_id = new_ids + prev_ids_number
            else:
                blob.previous_match = matched_blob
                matched_id = matched_blob.id
                
            blob.id = matched_id
            blob.write_text(self.blobs_remapped, str(blob.id))

    def apply_classification(self, classification_threshold):
        self.blobs_classified = self.image_blobs.copy()
        for blob in self.blobs:
            blob_class = blob.classify(classification_threshold)
            blob.write_text(self.blobs_classified, str(blob_class) + " " + str(blob.classification_score()))

    def apply_object_recognition(self, edge_threshold, edge_adaptation):
        self.blobs_detected = self.image.copy()
        for blob in self.blobs:
            if blob.detect(edge_threshold, edge_adaptation):
                name = "True"
                self.blobs_detected = np.where(blob.mask > 0, 255, self.blobs_detected)
            else:
                name = "False"
            blob.write_text(self.blobs_detected, name+" " + str(blob.edge_score()))
        return

        self.blobs_detected = self.image_blobs.copy()
        for blob in self.blobs:
            if blob.detect(edge_threshold):
                name = "True"
            else:
                name = "False"
            blob.write_text(self.blobs_detected, name+" " + str(blob.edge_score()))

    def generate_output(self):
        for blob in self.blobs:
            cv2.drawContours(self.image_output, blob.contours, -1, blob.color, 1)
            #blob_frame[blob.mask > 0] = blob.color

    def write_text_output(self, csv_writer, frame_index):
        csv_writer.writerow([frame_index, len(self.blobs)])
        for blob in self.blobs:
            csv_writer.writerow(blob.attributes())

class Background(Displayable):
    '''
        Class background providing operations to obtain the background
    '''

    def __init__(self, input_video_path=None, interpolation=None, frames_n=None, image=None):
        '''
            Estimates the background of the given video capture by using the interpolation function and n frames
            Returning a matrix of float64
        '''
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blind = None

        if input_video_path is not None:
            # Loading Video
            cap = cv2.VideoCapture(input_video_path)
            bg = []
            idx = 0
            # Initialize the background image
            while(cap.isOpened() and idx < frames_n):
                ret, frame = cap.read()
                if ret and not frame is None:
                    # frame = frame.astype(float)
                    # Getting all first n images
                    bg.append(frame[:,:,0])
                    idx += 1
                else:
                    break
            cap.release()

            self.image = interpolation(np.stack(bg, axis=0), axis=0).astype(np.uint8)
            self.subtraction = None
            self.mask_raw = None
            self.mask_refined = None
            self.name = "{}_{}".format(frames_n, interpolation.__name__)
        elif image is not None:
            self.image = image
        else:
            raise ValueError("Either an input_video_path or an image have to be specified")

    def __str__(self):
        return self.name

    def update_blind(self, frame, alpha):
        '''
            Method that conmputes the background via blind updating
        '''
        return (self.image * (1 - alpha) + frame.image * alpha).astype(np.uint8)

    def update_selective(self, frame, threshold, distance, alpha, morph_ops):
        '''
            Method that computes the background via selective updating
        '''
        self.subtraction = self.subtract_frame(frame.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)
        self.mask_refined = morph_ops.apply(self.mask_raw)
        
        self.blind = self.update_blind(frame, alpha)
        return np.where(self.mask_refined == 0, self.blind, self.image).astype(np.uint8)
        
    def subtract_frame(self, frame, distance):
        '''
            Computes the Background subtraction (distance(frame,background)) and returns a matrix of boolean
        '''
        return distance(frame, self.image)
