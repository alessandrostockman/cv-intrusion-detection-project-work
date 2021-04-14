import os
from copy import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt

from intrusiondetection.blob import Blob

class Displayable:

    def display_row(self, cols):
        '''
            Auxiliary method for displaying a set of images by a lsit of attribute names
        '''
        cols_number = len(cols)
        plt.figure(figsize=(5 * cols_number, 2.5 * cols_number))

        for col_index, image_dict in enumerate(cols):
            plt.subplot(1, cols_number, col_index+1)
            img = getattr(self, image_dict['key'])
            plt.axis('off')
            if image_dict['title'] is not None:
                plt.title(image_dict['title'])

            if img.shape[-1] == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)


    def display(self, keys, title=None, show=True, cols_number=1):
        '''
            Auxiliary method for displaying the image by attribute name
        '''
        if show:
            plt.figure(figsize=(5 * cols_number, 2.5 * cols_number))

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
        self.image_output = self.image_triple_channel.copy()
        
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_labeled = None
        self.blobs_remapped = None
        self.blobs_classified = None
        self.blobs_detected = None

        self.blobs = []
        self.max_blob_id = 0

    def intrusion_detection(self, params, bg, previous_blobs=[], blob_base_id=0):
        '''
            Performs the whole intrusion detection algorithm and computes the outputs
        '''
        self.apply_change_detection(bg, params.threshold, params.distance)
        self.apply_morphology_operators(params.morph_ops)
        self.apply_blob_analysis(previous_blobs, params, base_id=blob_base_id)
    
    def apply_change_detection(self, background, threshold, distance):
        '''
            Computes the difference between the frame and the background (computed via Selective update) and computes a binary mask of type np.uint8 
        '''
        self.subtraction = background.subtract_frame(self.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)

    def apply_morphology_operators(self, morph_ops):
        '''
            Applies binary morphology operators to the raw mask
        '''
        self.mask_refined = morph_ops.apply(self.mask_raw)

    def apply_blob_analysis(self, previous_blobs, params, base_id=0):
        '''
            Shorthand for performing the various blob analysis tasks
        '''
        self.apply_blob_labeling(create_output='blobs_labeled' in params.output_streams['foreground'])
        self.apply_blob_remapping(previous_blobs, params.similarity_threshold, base_id=base_id, create_output='blobs_remapped' in params.output_streams['foreground'])
        self.apply_classification(params.classification_threshold, create_output='blobs_classified' in params.output_streams['foreground'])
        self.apply_object_recognition(params.edge_threshold, params.edge_adaptation, create_output='blobs_detected' in params.output_streams['foreground'])
        self.generate_graphical_output()

    def apply_blob_labeling(self, create_output=False):
        '''
            Creates a Blob object for every blob found using OpenCV's connectedComponents function, performing thus BBDT algorithm with 8-way connectivity
        '''
        num_labels, labels = cv2.connectedComponents(self.mask_refined)
        self.blobs_labeled = self.image_triple_channel.copy()

        if num_labels > 1:
            # There is at least one blob detected
            for curr_label in range(1, num_labels):
                blob_mask = np.where(labels == curr_label, 255, 0).astype(np.uint8)
                blob = Blob(curr_label, blob_mask, self.image)
                self.blobs.append(blob)

                if create_output:
                    self.blobs_labeled[blob_mask > 0] = blob.color_palette[curr_label]
                    blob.write_text(self.blobs_labeled, blob.label)

    def apply_blob_remapping(self, previous_blobs, similarity_threshold, base_id=0, create_output=False):
        '''
            Generates a unique ID for every blob searching for correspondences in the previous_blobs list by computing a similarity score
        '''
        candidate_blobs = copy(previous_blobs)
        self.blobs_remapped = self.image_triple_channel.copy()
        
        for blob in self.blobs:
            matched_blob = blob.search_matching_blob(candidate_blobs, similarity_threshold)

            # When no match is found a new identifier is assigned
            if matched_blob is None:
                base_id += 1
                matched_id = base_id
            else:
                blob.previous_match = matched_blob
                matched_id = matched_blob.id
                
            blob.id = matched_id
            # Update max blob id (will be used as base_id for next iteration) 
            self.max_blob_id = max(self.max_blob_id, blob.id)

            if create_output:
                self.blobs_remapped[blob.mask > 0] = blob.color_palette[blob.id]
                blob.write_text(self.blobs_remapped, blob.id)

    def apply_classification(self, classification_threshold, create_output=False):
        '''
            Computes a classification score for each blob and assign them either a PERSON or OBJECT class
        '''
        self.blobs_classified = self.image_triple_channel.copy()
        for blob in self.blobs:
            blob_class = blob.classify(classification_threshold)

            if create_output:
                blob.write_text(self.blobs_classified, str(blob_class) + " - " + str(blob.classification_score()))

    def apply_object_recognition(self, edge_threshold, edge_adaptation, create_output=False):
        '''
            Computes an edge score for each blob and determines whether the object is actually present or corresponds to the absence of an object in the background
        '''
        self.blobs_detected = self.image_triple_channel.copy()
        for blob in self.blobs:
            presence = blob.detect(edge_threshold, edge_adaptation)
            
            if create_output:
                if presence:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                self.blobs_detected[blob.mask > 0] = color
                blob.write_text(self.blobs_detected, blob.edge_score())

    def generate_graphical_output(self):
        '''
            Computes the final graphical output
        '''
        for blob in self.blobs:
            cv2.drawContours(self.image_output, blob.contours, -1, blob.color, 1)

    def generate_text_output(self, csv_writer, frame_index):
        '''
            Computes the final text output
        '''
        csv_writer.writerow([frame_index, len(self.blobs)])
        for blob in self.blobs:
            csv_writer.writerow(blob.attributes())

class Background(Displayable):
    '''
        Class background providing operations to obtain the background
    '''

    def __init__(self, input_video_path=None, interpolation=None, frames_n=None, image=None):
        '''
            Initializes the background image either by using a video source and an interpolation method or by assigning directly an image 
        '''
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blind = None

        if input_video_path is not None and interpolation is not None and frames_n is not None:
            if not os.path.isfile(input_video_path):
                raise IOError("Video " + input_video_path + " doesn't exist")

            # The background image is computed using the interpolation function over the first frames_n number of frames of the given video
            cap = cv2.VideoCapture(input_video_path)
            bg = []
            idx = 0
            while(cap.isOpened() and idx < frames_n):
                ret, frame = cap.read()
                if ret and not frame is None:
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
            Returns the blending of the background image and the given frame weighted by the adaptation rate alpha
        '''
        return (self.image * (1 - alpha) + frame.image * alpha).astype(np.uint8)

    def update_selective(self, frame, threshold, distance, alpha, morph_ops):
        '''
            Returns the blending of the background image and the given frame weighted by the adaptation rate alpha selectively on background pixels
        '''
        self.subtraction = self.subtract_frame(frame.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)
        self.mask_refined = morph_ops.apply(self.mask_raw)
        
        self.blind = self.update_blind(frame, alpha)
        return np.where(self.mask_refined == 0, self.blind, self.image).astype(np.uint8)
        
    def subtract_frame(self, frame, distance):
        '''
            Returns the background subtraction using the given distance function
        '''
        return distance(frame, self.image)
