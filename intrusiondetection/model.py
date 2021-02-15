import csv
import numpy as np
import cv2
from enum import Enum
from copy import copy

class Video:
    '''
    Class Video defining the video that will be written in output
    '''
    def __init__(self, params):
        #String for creating the name of the output video 
        video_output_streams = ['blobs_contours', 'blobs_filled', 'mask_refined', 'subtraction', 'mask_raw', 'mask_refined']
        #String for creating the name of the background video
        video_bg_output_streams = ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']

        # outputs = ['mask_raw','mask_refined','bg_image','blob_cont','blob_fill','bg_mask_raw','bg_mask_refined'] TODO

        self.frame_index = 0
        self.params = params
        self.cap = cv2.VideoCapture(self.params.input_video)

        #Width
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #Height
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        #creating the names
        self.outputs = {key: self.create_output_stream(w, h, fps, str(self.params) + "_"+key+".avi") for key in video_output_streams}
        self.bg_outputs = {key: self.create_output_stream(w, h, fps, str(self.params) + "_bg_"+key+".avi") for key in video_bg_output_streams}
        

    def intrusion_detection(self):
        '''
        Method that computes the change detection:
        Compuetes the background via selective update;
        applies the change detection;
        applies morphology;
        applies blob analysis;
        For every blob we write the classification of the blob. 
        Modifies the output to write them on the output stream.
        '''
        try:
            csv_file = open(self.params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            bg = self.params.background
            blobs = []
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                self.params.frame = frame
                if not ret or frame is None:
                    print("Computation finished! Video saved in {}".format(self.params.output_video))
                    break

                fr = Frame(frame, bg)
                bg.image = bg.update_selective(fr, self.params.background_threshold, self.params.background_distance, self.params.alpha, self.params.background_morph_ops)
                fr.apply_change_detection(self.params.threshold, self.params.distance)
                fr.apply_morphology_operators(self.params.morph_ops)
                fr.apply_blob_analysis(blobs)
                blobs = fr.blobs #TODO Meglio

                #TODO Refactor
                for blob in blobs:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(fr.blobs_contours, str(blob), (blob.cx, blob.cy), font, .2, (255,255,255), 1, cv2.LINE_AA)

                #TODO effettivamente non so che fanno questi due cicli, cambiano il tipo di output da scrivere ?
                for key, out in self.outputs.items():
                    x = getattr(fr, key)
                    if x.shape[-1] != 3:
                        x = np.tile(x[:,:,np.newaxis], 3)
                    if x.dtype != np.uint8:
                        x = x.astype(np.uint8)

                    out.write(x)

                #TODO Migliorare?
                for key, out in self.bg_outputs.items():
                    x = getattr(bg, key)
                    if x.shape[-1] != 3:
                        x = np.tile(x[:,:,np.newaxis], 3)
                    if x.dtype != np.uint8:
                        x = x.astype(np.uint8)

                    out.write(x)

                fr.write_text_output(csv_writer, frame_index=self.frame_index)
                self.frame_index += 1
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

class Frame:
    '''
        Class Frame containing the methods that needs to be applied on the image frame

    '''
    def __init__(self, image, background):
        self.image = image[:,:,0]
        self.image_triple_channel = image
        self.background = background
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_contours = None
        self.blobs_filled = None

        self.blobs = []
    
    def apply_change_detection(self, threshold, distance):
        '''
            Computes the difference between the frame and the background (computed via Selective update) and computes a binary mask of type np.uint8 
        '''
        self.subtraction = self.background.subtract_frame(self.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)

    def apply_morphology_operators(self, morph_ops):
        '''
            Computes the binary morphology on the raw mas and assigns it on the mask_refined of the class
        '''
        self.mask_refined = morph_ops.apply(self.mask_raw)

    def apply_blob_analysis(self, previous_blobs, similarity_threshold=5000):
        '''
            Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        num_labels, labels = cv2.connectedComponents(self.mask_refined)

        #No blobs detected
        if num_labels <= 1:
            self.blobs = []
            self.blobs_filled = self.image
            self.blobs_contours = self.image
            return
        
        new_labels = 0
        blobs = []

        candidate_blobs = copy(previous_blobs)
        #operating on each blob found in the frame
        for curr_label in range(1, num_labels):
            blob_mask = np.where(labels == curr_label, 255, 0).astype(np.uint8)
            
            blob = Blob(self.image, blob_mask)
            
            #blob is not created due to noise
            if blob.is_valid:
                matched_label = blob.search_matching_blob(candidate_blobs, similarity_threshold)

                #if we canno't match the blob to a previous blob then it means that we need to associate a new label on the detected blob
                if matched_label is None:
                    prev_labels_number = max([x.label for x in previous_blobs], default=0)
                    new_labels += 1
                    matched_label = new_labels + prev_labels_number
                    print("New object found:", matched_label, "Prev Label:", prev_labels_number, "New:", new_labels)
                    
                blob.label = matched_label
                blobs.append(blob)

        blob_frame = self.image_triple_channel.copy()
        cont_frame = self.image_triple_channel.copy()
        #TODO Trovare un modo per fare i colori
        color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        for index, blob in enumerate(blobs):
            #color = color_palette[blob.label]
            color = color_palette[0]

            #TODO migliorare e bordi più sottili
            countours_frame = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(countours_frame, blob.contours, -1, color, 1)
            cont_frame[np.sum(countours_frame, axis=-1) > 0] = color

            blob_frame[blob.mask > 0] = color

        self.blobs = blobs
        self.blobs_filled = blob_frame
        self.blobs_contours = cont_frame

    def write_text_output(self, csv_writer, frame_index):
        csv_writer.writerow([frame_index, len(self.blobs)])
        for blob in self.blobs:
            csv_writer.writerow(blob.attributes())

class Background:
    '''
        Class background providing operations to obtain the background
    '''
    def __init__(self, input_video_path, interpolation, frames_n=100):
        '''
            Estimates the background of the given video capture by using the interpolation function and n frames
            Returning a matrix of float64
        '''
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

        #TODO Temp
        self.blind = None
        self.init = self.image.copy()

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
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        return mask

class Blob:

    def __init__(self, frame, mask):
        self.label = None
        self.contours = self.parse_contours(mask)
        self.main_contours = self.contours[0]
        self.image = frame
        self.mask = mask

        #TODO Controllare se le feature si possono calcolare in modi migliori

        self.area = cv2.contourArea(self.main_contours)
        self.perimeter = cv2.arcLength(self.main_contours, True)
        moments = cv2.moments(self.main_contours)
        
        #TODO Remove
        self.broken = False
        if moments['m00'] == 0: 
            self.broken = True
            moments['m00'] = 1
        
        #Obtaining the barycentre of the blob
        self.cx = int(moments['m10']/moments['m00'])
        self.cy = int(moments['m01']/moments['m00'])

        sum = len(self.main_contours)

        #Calculating the derivatives to obtain the gradient value to verify if the object is a true object or a fake one
        val = 0
        x_max, y_max = frame.shape
        for coord in self.main_contours:
            y, x = coord[0][0], coord[0][1]

            if y <= 0 or y >= y_max - 1 or x <= 0 or x >= x_max - 1:
                dx = 0
                dy = 0
            else:
                dx = self.i4y(frame, x, y+1) - self.i4y(frame, x, y-1)
                dy = self.i4x(frame, x+1, y) - self.i4x(frame, x-1, y)

            val += max(abs(dx), abs(dy))
        self.edge_strength = val / sum
        
        #not considering blob which area is below 500 pixel cause they are due to noise 
        #TODO Maybe create a list of thresholds
        self.is_valid = self.area > 500
        #detecting true blob from fake one in base of the gradient of the value of the edge
        self.is_present = self.edge_strength > 60
        #Distinguish wether a blob is a person or an object in base of the are of his blob 
        #TODO crare una lista che contiene i vari threshold ?
        self.blob_class = BlobClass.PERSON if self.area > 1000 else BlobClass.OBJECT

    def __str__(self):
        name = ""
        if self.is_present:
            name = str(self.blob_class)
        else:
            name = "FAKE"
        return str(self.label) + " " + str(self.area) + " " + name

    #TODO dobbiamo toglierlo sto 1/4 o no ?
    def i4x(self, frame, i, j):
        '''
            Smooth derivative along x
        '''
        return 1/4 * frame[i, j-1] + 2 * frame[i, j] + frame[i, j+1]

    def i4y(self, frame, i, j):
        '''
            Smooth derivative along y
        '''
        return 1/4 * frame[i-1, j] + 2 * frame[i, j] + frame[i+1, j]

    def attributes(self):
        return [self.label, self.area, self.perimeter, self.blob_class]

    def parse_contours(self, image):
        ret = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours

    def search_matching_blob(self, candidate_blobs, threshold):
        '''
            Detcting matching blobs using the dissimilarity method shown below
        '''
        label = None
        if len(candidate_blobs) > 0:
            best_blob = None
            best_dissimilarity = 1000000
            best_index = -1
            for index, candidate_blob in enumerate(candidate_blobs):
                dissimilarity = self.dissimilarity(candidate_blob)
                if dissimilarity < threshold and dissimilarity < best_dissimilarity:
                    best_blob = candidate_blob
                    best_dissimilarity = dissimilarity
                    best_index = index

            if best_blob is not None:
                label = best_blob.label
                candidate_blobs.pop(best_index)
        return label

    def dissimilarity(self, other):
        '''
            Calculating the dissimilarity of two blobs, as lower the dissimilarity as more likely the two blobs represent the same one in two different frames
        '''
        area_diff = abs(other.area - self.area)
        barycenter_diff = abs((other.cx - self.cx) + (other.cy - self.cy))
        return area_diff + barycenter_diff

class BlobClass(Enum):
    '''
        Enum to distinguish the type of the blob
    '''
    PERSON = 1
    OBJECT = 2

    def __str__(self):
        return self.name