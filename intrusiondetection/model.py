import csv
import numpy as np
import cv2
from enum import Enum

class Video:

    def __init__(self, params):
        video_output_streams = ['subtraction', 'mask_raw', 'mask_refined', 'image']
        video_bg_output_streams = ['subtraction', 'mask_raw', 'mask_refined', 'image', 'blind']

        # outputs = ['mask_raw','mask_refined','bg_image','blob_cont','blob_fill','bg_mask_raw','bg_mask_refined'] TODO

        self.frame_index = 0
        self.params = params
        self.cap = cv2.VideoCapture(self.params.input_video)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.outputs = {key: self.create_output_stream(w, h, fps, str(self.params) + "_"+key+".avi") for key in video_output_streams}
        self.bg_outputs = {key: self.create_output_stream(w, h, fps, str(self.params) + "_bg_"+key+".avi") for key in video_bg_output_streams}
        

    def intrusion_detection(self):
        try:
            csv_file = open(self.params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            bg = self.params.background
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                self.params.frame = frame
                if not ret or frame is None:
                    print("Computation finished! Video saved in {}".format(self.params.output_video))
                    break

                fr = Frame(frame[:,:,0], bg)
                bg.image = bg.update_selective(fr, self.params.threshold, self.params.distance, self.params.alpha, self.params.background_morph_ops)
                fr.apply_change_detection(self.params.threshold, self.params.distance)
                fr.apply_morphology_operators(self.params.morph_ops)
                fr.apply_blob_analysis()

                for key, out in self.outputs.items():
                    x = getattr(fr, key)
                    out.write(np.tile(x[:,:,np.newaxis], 3))

                #TODO Migliorare?
                for key, out in self.bg_outputs.items():
                    x = getattr(bg, key)
                    out.write(np.tile(x[:,:,np.newaxis], 3))

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

    def __init__(self, image, background):
        self.image = image
        self.background = background
        self.subtraction = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_contours = None
        self.blobs_filled = None

        self.blobs = []

    def apply_change_detection(self, threshold, distance):
        self.subtraction = self.background.subtract_frame(self.image, distance).astype(np.uint8)
        self.mask_raw = np.where(self.subtraction > threshold, 255, 0).astype(np.uint8)

    def apply_morphology_operators(self, morph_ops):
        self.mask_refined = morph_ops.apply(self.mask_raw.copy())

    def apply_blob_analysis(self):

        '''Detects the blobs and creates them. Returns the blobs (as dictionaries) and the respective frames with the contour drawn on it
        ''' 
        num_labels, labels = cv2.connectedComponents(self.mask_refined)

        if num_labels <= 1:
            return
        
        blobs = []
        for curr_label in range(1, num_labels):
            blob_image = np.where(labels == curr_label, 255, 0).astype(np.uint8)
            blobs.append(Blob(curr_label, blob_image))

        self.blobs = blobs
        self.blobs_filled = None
        self.blobs_contours = None

    def write_text_output(self, csv_writer, frame_index):
        csv_writer.writerow([frame_index, len(self.blobs)])
        for blob in self.blobs:
            csv_writer.writerow(blob.attributes())

class Background:
    
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

    def __str__(self):
        return self.name

    def update_blind(self, frame, alpha):
        return (self.image * (1 - alpha) + frame.image * alpha).astype(np.uint8)

    def update_selective(self, frame, threshold, distance, alpha, morph_ops):
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
        for op in self.get():
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        return mask

class Blob:

    def parse_contours(self, thresh):
        ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) > 2:
            _, contours, hierarchy = ret
        else:
            contours, hierarchy = ret

        return contours

    def __init__(self, label, image):
        cnts = self.parse_contours(image)
        cnts = cnts[0]
        self.label = label
        self.blob_class = BlobClass.PERSON

        self.area = cv2.contourArea(cnts)
        self.perimeter = cv2.arcLength(cnts, True)
        M = cv2.moments(cnts)
        if M['m00'] == 0:
            return # TODO ???
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])

    def attributes(self):
        return [self.label, self.area, self.perimeter, self.blob_class]

        

class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2