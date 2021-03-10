import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from enum import Enum
from copy import copy
import math

from intrusiondetection.utility import hue_to_rgb

#TODO Decompose in different files

class Video:
    '''
    Class Video defining the video that will be written in output
    '''
    def __init__(self, input_video_path):
        self.frame_index = 0
        self.frames = []
        self.backgrounds = []
        self.cap = cv2.VideoCapture(input_video_path)

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.load_video()
        
    def load_video(self):
        while self.cap.isOpened():
            ret, frame_image = self.cap.read()
            if not ret or frame_image is None:
                break

            self.frames.append(Frame(frame_image))

    def process_backgrounds(self, update_mode, initial_background, alpha, threshold=None, distance=None, morph_ops=None):
        """
            Method used only for demonstration purposes
        """
        bg = initial_background
        backgrounds = [bg]
        for fr in self.frames:
            if update_mode == BackgroundMethod.SELECTIVE:
                bg_image = bg.update_selective(fr, threshold, distance, alpha, morph_ops)
            elif update_mode == BackgroundMethod.BLIND:
                bg_image = bg.update_blind(fr, alpha)
            else:
                raise ValueError("update_mode must be a BackgroundMethod")


            bg = Background(image=bg_image)
            backgrounds.append(bg)
        return backgrounds

    def intrusion_detection(self, params, initial_background):
        '''
        Method that computes the change detection:
        Compuetes the background via selective update;
        applies the change detection;
        applies morphology;
        applies blob analysis;
        For every blob we write the classification of the blob. 
        Modifies the output to write them on the output stream.
        '''

        self.outputs = {output_type: {
            key: self.create_output_stream(self.w, self.h, self.fps, str(params) + "_" + output_type + "_" + key + ".avi") for key in outputs
        } for output_type, outputs in params.output_streams.items()}

        try:
            csv_file = open(params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            prev_fr = None
            prev_bg = initial_background

            for fr in self.frames:
                prev_blobs = []
                if prev_fr is not None:
                    prev_blobs = prev_fr.blobs

                bg_image = prev_bg.update_selective(fr, params.background_threshold, params.background_distance, params.background_alpha, params.background_morph_ops)
                bg = Background(image=bg_image)

                fr.intrusion_detection(params, bg, prev_blobs)

                for output_type, outputs in self.outputs.items():
                    obj = fr if output_type == 'foreground' else prev_bg
                    for key, out in outputs.items():
                        output_image = getattr(obj, key, None)

                        if output_image is not None:
                            if output_image.shape[-1] != 3:
                                output_image = np.tile(output_image[:,:,np.newaxis], 3)
                            if output_image.dtype != np.uint8:
                                output_image = output_image.astype(np.uint8)                        
                            out.write(output_image)

                fr.write_text_output(csv_writer, frame_index=self.frame_index)
                self.frame_index += 1
                prev_fr = fr
                prev_bg = bg
        finally:
            csv_file.close()
    
    def create_output_stream(self, w, h, fps, output_video_path):
        '''
            Creates a video writer reference for output_video_path
        '''
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w,  h))

        return out

from copy import copy

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
                plt.imshow(img)
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
            plt.imshow(img)
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
        self.apply_blob_analysis(previous_blobs, params.similarity_threshold, params.classification_threshold, params.edge_threshold)
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
        self.mask_refined, m = morph_ops.apply(self.mask_raw)
        i = 0
        for mask in m:
            setattr(self, 'mask_'+str(i), mask)
            i += 1

    def apply_blob_analysis(self, previous_blobs, similarity_threshold, classification_threshold, edge_threshold):
        self.apply_blob_labeling()
        self.apply_blob_remapping(previous_blobs, similarity_threshold)
        self.apply_classification(classification_threshold)
        self.apply_object_recognition(edge_threshold)
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

    def apply_object_recognition(self, edge_threshold):
        self.blobs_detected = self.image.copy()
        for blob in self.blobs:
            if blob.detect(edge_threshold):
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
        self.mask_refined, m = morph_ops.apply(self.mask_raw)
        for i, mask in enumerate(m):
            setattr(self, 'mask_'+str(i), mask)
        
        self.blind = self.update_blind(frame, alpha)
        return np.where(self.mask_refined == 0, self.blind, self.image).astype(np.uint8)
        
    def subtract_frame(self, frame, distance):
        '''
            Computes the Background subtraction (distance(frame,background)) and returns a matrix of boolean
        '''
        return distance(frame, self.image)

class MorphOp:
    '''
        Class MorphOp that defines the morphology operation that will be applied
    '''
    def __init__(self, operation_type, kernel_size, kernel_shape=None, iterations=1):
        self.operation_type = operation_type
        self.kernel_size = kernel_size
        self.iterations = iterations
        if kernel_shape is None:
            self.kernel = np.ones(kernel_size, dtype=np.uint8)
        else:
            self.kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

    def __str__(self):
        name = ""
        if self.operation_type == cv2.MORPH_CLOSE:
            name += "C"
        elif self.operation_type == cv2.MORPH_OPEN:
            name += "O"
        elif self.operation_type == cv2.MORPH_DILATE:
            name += "D"
        elif self.operation_type == cv2.MORPH_ERODE:
            name += "E"

        x, y = self.kernel_size
        name += str(x) + "x" + str(y)
        return name
class MorphOpsSet:
    def __init__(self, *ops):
        self.ops = ops

    def __str__(self):
        return "".join(str(x) for x in self.ops)

    def get(self):
        return self.ops

    def apply(self, mask):
        '''
            multiple iterations of closing or opening means that we apply n°iterations-times of Dilate + n°iterations-times of Erosion or vice-versa
        '''
        masks = [] #TODO Remove
        for op in self.get():
            kernel_x, kernel_y = op.kernel.shape
            mask=cv2.copyMakeBorder(mask, kernel_y, kernel_y, kernel_x, kernel_x,
                borderType=cv2.BORDER_CONSTANT, value=0)
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
            mask = mask[kernel_y:-kernel_y, kernel_x:-kernel_x]

            masks.append(mask)
        return mask, masks

class BlobClass(Enum):
    '''
        Enum to distinguish the type of the blob
    '''
    PERSON = 1
    OBJECT = 2
    FAKE = 3

    def __str__(self):
        return self.name

class BackgroundMethod(Enum):
    BLIND = 1
    SELECTIVE = 2

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

