import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from enum import Enum
from copy import copy

class Video:
    '''
    Class Video defining the video that will be written in output
    '''
    def __init__(self, input_video_path):
        self.frame_index = 0
        self.frames = []
        self.backgrounds = []
        self.cap = cv2.VideoCapture(input_video_path)

        #Width
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #Height
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fps
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.load_video()
        
    def load_video(self):
        while self.cap.isOpened():
            ret, frame_image = self.cap.read()
            if not ret or frame_image is None:
                break

            self.frames.append(Frame(frame_image))

    def process_backgrounds(self, update_mode, initial_background, alpha, threshold=None, distance=None, morph_ops=None):
        bg = initial_background
        backgrounds = [bg]
        for fr in self.frames:
            if update_mode == 'selective':
                bg_image = bg.update_selective(fr, threshold, distance, alpha, morph_ops)
            else:
                bg_image = bg.update_blind(fr, alpha)

            bg = Background(image=bg_image)
            backgrounds.append(bg)
        return backgrounds

    def intrusion_detection(self, params, backgrounds):
        '''
        Method that computes the change detection:
        Compuetes the background via selective update;
        applies the change detection;
        applies morphology;
        applies blob analysis;
        For every blob we write the classification of the blob. 
        Modifies the output to write them on the output stream.
        '''

        output_streams = {
            'foreground': ['blobs_contours', 'blobs_filled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined'],
            'background': ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']
        }

        # outputs = ['mask_raw','mask_refined','bg_image','blob_cont','blob_fill','bg_mask_raw','bg_mask_refined'] TODO
        #creating the output streams
        self.outputs = {output_type: {
            key: self.create_output_stream(self.w, self.h, self.fps, str(params) + "_" + output_type + "_" + key + ".avi") for key in outputs
        } for output_type, outputs in output_streams.items()}

        try:
            csv_file = open(params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            prev_fr = None

            for fr in self.frames:
                prev_blobs = []
                bg = backgrounds[self.frame_index]
                if prev_fr is not None:
                    prev_blobs = prev_fr.blobs

                fr.intrusion_detection(params, bg, prev_blobs)

                for output_type, outputs in self.outputs.items():
                    obj = fr if output_type == 'foreground' else bg
                    for key, out in outputs.items():
                        x = getattr(obj, key)
                        if x.shape[-1] != 3:
                            x = np.tile(x[:,:,np.newaxis], 3)
                        if x.dtype != np.uint8:
                            x = x.astype(np.uint8)                        
                        out.write(x)

                fr.write_text_output(csv_writer, frame_index=self.frame_index)
                self.frame_index += 1
                prev_fr = fr
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
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_contours = self.image_triple_channel
        self.blobs_filled = self.image_triple_channel

        self.blobs = []

    def intrusion_detection(self, params, bg, previous_blobs):
        self.apply_change_detection(bg, params.threshold, params.distance)
        self.apply_morphology_operators(params.morph_ops)
        self.apply_blob_analysis(previous_blobs, 5000) #TODO Threshold

        #TODO Refactor
        print(self.blobs_contours.shape, self.blobs_contours.dtype)
        for blob in previous_blobs:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.blobs_contours, str(blob), (blob.cx, blob.cy), font, .2, (255,255,255), 1, cv2.LINE_AA)
    
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

    def apply_blob_analysis(self, previous_blobs, similarity_threshold):
        '''
            Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        num_labels, labels = cv2.connectedComponents(self.mask_refined)

        #No blobs detected
        if num_labels <= 1:
            self.blobs = []
            self.blobs_filled = self.image
            self.blobs_labeled = self.image
            self.blobs_remapped = self.image
            self.blobs_classified = self.image
            self.blobs_refined = self.image
            #self.blobs_contours = self.image
            return
        
        new_labels = 0
        blobs = []

        candidate_blobs = copy(previous_blobs)
        #operating on each blob found in the frame
        for curr_label in range(1, num_labels):
            blob_mask = np.where(labels == curr_label, 255, 0).astype(np.uint8)
            
            blob = Blob(self.image, blob_mask, 100, 2000)
            
            #blob is not created due to noise
            if blob.is_valid:
                matched_label = blob.search_matching_blob(candidate_blobs, similarity_threshold)

                #if we canno't match the blob to a previous blob then it means that we need to associate a new label on the detected blob
                if matched_label is None:
                    prev_labels_number = max([x.label for x in previous_blobs], default=0)
                    new_labels += 1
                    matched_label = new_labels + prev_labels_number
                    
                blob.label = matched_label
                blobs.append(blob)

        blob_frame = self.image_triple_channel.copy()
        cont_frame = self.image_triple_channel.copy()
        #TODO Trovare un modo per fare i colori
        color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        for index, blob in enumerate(blobs):
            #color = color_palette[blob.label]
            color = color_palette[0] #TODO Color based on classification

            cv2.drawContours(cont_frame, blob.contours, -1, color, 1)
            blob_frame[blob.mask > 0] = color

        self.blobs = blobs
        self.blobs_filled = blob_frame
        self.blobs_labeled = blob_frame
        self.blobs_remapped = blob_frame
        self.blobs_classified = blob_frame
        self.blobs_refined = blob_frame
        self.blobs_contours = cont_frame

    def apply_blob_labeling(self, colored_output=False):
        labels_num, self.blob_labeled = cv2.connectedComponents(self.mask_refined)

        self.x = labels * color

    def apply_blob_linkage(self, labels_num, dissimilarity_threshold):
        new_labels = 0
        blobs = []
        blobs = []
        for curr_label in range(1, labels_num):
            blob_mask = np.where(labels == curr_label, 255, 0).astype(np.uint8)
            blob = Blob(self.image, blob_mask, 100, 2000) #TODO Spostare i threhsold
            
            #blob is not created due to noise
            if blob.is_valid:
                matched_label = blob.search_matching_blob(candidate_blobs, similarity_threshold)

                #if we canno't match the blob to a previous blob then it means that we need to associate a new label on the detected blob
                if matched_label is None:
                    prev_labels_number = max([x.label for x in previous_blobs], default=0)
                    new_labels += 1
                    matched_label = new_labels + prev_labels_number
                    
                blob.label = matched_label
                blobs.append(blob)
        self.blobs = blobs

    def apply_classification(self, classification_threshold):
        pass

    def apply_object_recognition(self, edge_threshold):
        pass

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

        #TODO Refactor
        mask1 = np.zeros(self.mask_refined.shape, dtype=np.uint8)
        mask2 = np.zeros(self.mask_refined.shape, dtype=np.uint8)
        mask1[self.mask_refined == 0] = 1
        mask2[self.mask_refined != 0] = 1

        self.blind = self.update_blind(frame, alpha)
        new = self.blind * mask1 + self.image * mask2
        return new.astype(np.uint8)
        
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
        for op in self.get():
            kernel_x, kernel_y = op.kernel.shape
            mask=cv2.copyMakeBorder(mask, kernel_y, kernel_y, kernel_x, kernel_x,
                borderType=cv2.BORDER_CONSTANT, value=0)
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
            mask = mask[kernel_y:-kernel_y, kernel_x:-kernel_x]
        return mask

class Blob:

    def __init__(self, frame, mask, edge_threshold, classification_threshold):
        self.label = None
        self.contours = self.parse_contours(mask)
        self.main_contours = self.contours[0]
        self.image = frame
        self.mask = mask

        #TODO Controllare se le feature si possono calcolare in modi migliori

        self.area = cv2.contourArea(self.main_contours)
        self.perimeter = cv2.arcLength(self.main_contours, True)
        moments = cv2.moments(self.main_contours)
        
        #Obtaining the barycentre of the blob
        self.cx = int(moments['m10']/moments['m00'])
        self.cy = int(moments['m01']/moments['m00'])
        
        self.edge_strength = self.edge_score(frame)
        #not considering blob which area is below 500 pixel cause they are due to noise 
        self.is_valid = True#self.area > 500
        #detecting true blob from fake one in base of the gradient of the value of the edge
        self.is_present = self.edge_strength > edge_threshold
        #Distinguish wether a blob is a person or an object in base of the are of his blob 
        self.blob_class = BlobClass.PERSON if self.classification_score() > classification_threshold else BlobClass.OBJECT

    def __str__(self):
        name = ""
        if self.is_present:
            name = str(self.blob_class)
        else:
            name = "FAKE"
        return str(self.classification_score()) + " " + name

    def attributes(self):
        return [self.label, self.area, self.perimeter, self.blob_class]

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
        label = None
        if len(candidate_blobs) > 0:
            best_blob = None
            best_dissimilarity = 1000000
            best_index = -1
            for index, candidate_blob in enumerate(candidate_blobs):
                dissimilarity = self.dissimilarity_score(candidate_blob)
                if dissimilarity < dissimilarity_threshold and dissimilarity < best_dissimilarity:
                    best_blob = candidate_blob
                    best_dissimilarity = dissimilarity
                    best_index = index

            if best_blob is not None:
                label = best_blob.label
                candidate_blobs.pop(best_index)
        return label

    def dissimilarity_score(self, other):
        '''
            Calculating the dissimilarity of two blobs, as lower the dissimilarity as more likely the two blobs represent the same one in two different frames
        '''
        area_diff = abs(other.area - self.area)
        barycenter_diff = abs((other.cx - self.cx) + (other.cy - self.cy))
        return area_diff + barycenter_diff

    def classification_score(self):
        return self.area


    def edge_score(self, frame):

        sum = len(self.main_contours)

        #Calculating the derivatives to obtain the gradient value to verify if the object is a true object or a fake one
        
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

            window = frame[x-1:x+2,y-1:y+2]
            if window.shape == (3, 3):
                val += np.maximum(abs((window * mat_x).sum()), abs((mat_y * window).sum()))
        
        valA = self.tmp()

        if valA != val:
            print("XXX")
            
        return valA / sum

    #TODO: Remove
    def tmp(self, frame):
        valA = 0
        x_max, y_max = frame.shape
        for coord in self.main_contours:
            y, x = coord[0][0], coord[0][1]

            if y <= 0 or y >= y_max - 1 or x <= 0 or x >= x_max - 1:
                dx = 0
                dy = 0
            else:
                dx = self.i4y(frame, x, y+1) - self.i4y(frame, x, y-1)
                dy = self.i4x(frame, x+1, y) - self.i4x(frame, x-1, y)

            valA += max(abs(dx), abs(dy))
        return valA

    #TODO: Remove
    def i4x(self, frame, i, j):
        '''
            Smooth derivative along x
        '''
        return frame[i, j-1] + 2 * frame[i, j] + frame[i, j+1]

    #TODO: Remove
    def i4y(self, frame, i, j):
        '''
            Smooth derivative along y
        '''
        return frame[i-1, j] + 2 * frame[i, j] + frame[i+1, j]

class BlobClass(Enum):
    '''
        Enum to distinguish the type of the blob
    '''
    PERSON = 1
    OBJECT = 2

    def __str__(self):
        return self.name

class BackgroundMethod(Enum):
    BLIND = 1
    SELECTIVE = 2