import csv
import numpy as np
import cv2

class Frame:

    def __init__(self, image):
        self.image = image
        self.background = None
        self.mask_raw = None
        self.mask_refined = None
        self.blobs_contours = None
        self.blobs_filled = None

        self.blobs = []

    def apply_change_detection(self, background):
        x.update_selective_background(self.image)
        self.mask_raw = x.background_subtraction(self.image, x.parameters.adaptive_background)

    
    def update_selective_background(self, background):
        new_bg = background.copy()
        new_bg.background_subtraction(self.image)
        new_bg.apply_morphology_operators()

        mask1 = np.zeros(new_bg.foreground_refined.shape, dtype=int)
        mask2 = mask1.copy()
        mask1[binary_res == 0] = 1
        mask2[binary_res != 0] = 1
        new_bg.result = self.compute_blind_background(frame)[:,:,0] * mask1 + new_bg[:,:,0] * mask2
        self.parameters.adaptive_background = np.tile(new_bg[:,:,np.newaxis], 3)

    def apply_morphology_operators(self, morph_ops):
        mask = self.foreground_raw.copy()
        for op in morph_ops:
            mask = cv2.morphologyEx(mask, op.operation_type, op.kernel, iterations=op.iterations)
        self.mask_refined = mask

    def apply_blob_analysis(self):
        self.blobs = blobs
        self.blobs_contours = countours_frame
        self.blobs_filled = blob_frame

    def write_text_output(self, csv_writer, frame_index):
        csv_writer.writerow([frame_index, len(self.blobs)])
        for blob in self.blobs:
            csv_writer.writerow(blob.get_attributes())

class Video:

    def __init__(self, params):
        video_output_streams = ['foreground_raw', 'foreground_refined']

        self.frame_index = 0
        self.params = params
        self.cap = cv2.VideoCapture(self.params.input_video)
        self.outputs = {key: create_output_stream(cap, str(self.params) + "_"+key+".avi") for key in video_output_streams}
        

    def intrusion_detection(self):
        # out1 = self.create_output_stream(cap, str(self.params) + "_cd.avi") #mask_raw
        # out2 = self.create_output_stream(cap, str(self.params) + "_bm.avi") #mask_refined
        # out3 = self.create_output_stream(cap, str(self.params) + "_bg.avi") #background.frame
        # out4 = self.create_output_stream(cap, str(self.params) + "_cont.avi") #blob_cont
        # out5 = self.create_output_stream(cap, str(self.params) + "_blob.avi") #blob_fill
        # out6 = self.create_output_stream(cap, str(self.params) + "_sub.avi") #background.mask_raw
        # out7 = self.create_output_stream(cap, str(self.params) + "_mor.avi") #background.mask_refined
        try:
            csv_file = open(self.params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            while cap.isOpened():
                ret, frame = cap.read()
                self.params.frame = frame
                if not ret or frame is None:
                    print("Computation finished! Video saved in {}".format(self.params.output_video))
                    break

                f = Frame()
                f.background = f.update_background()
                f.apply_change_detection(f.background)
                f.apply_morphology_operators(self.params.morph_ops.get())
                f.apply_blob_analysis()

                bg = f.background
                for key, out in self.outputs:
                    #TODO if bg -> getattr(bg, key)
                    out.write(getattr(f, key))


                #frame1 = self.cd.apply(frame)
                #frame2 = self.bm.apply(frame1)
                #frame3_contours, frame3_blob, blobs = self.cc.apply(frame2)
                #append_text_output(idx, blobs, csv_writer)

                #out.write(frame3)
                #out1.write(np.tile(frame1.astype(np.uint8)[:,:,np.newaxis], 3) * 255)
                #out2.write(np.tile(frame2[:,:,np.newaxis], 3) * 255)
                #out3.write(self.params.adaptive_background.astype(np.uint8))
                #out4.write(frame3_contours)
                #out5.write(frame3_blob)
                #out6.write(np.tile(self.params.sub[:,:,np.newaxis], 3))
                #out7.write(np.tile(self.params.mor[:,:,np.newaxis], 3))

                f.write_text_output(csv_writer, frame_index=self.frame_index)
                self.frame_index += 1
        finally:
            csv_file.close()
    
    def create_output_stream(self, template_cap, output_video_path):
        '''
            Creates a video writer reference for output_video_path
        '''
        # Getting original video params
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w,  h))

        return out

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

    def get(self):
        return self.ops

    def __str__(self):
        return "".join(str(x) for x in self.ops)

class Blob:

    def __init__(self, label, cnts):
        cnts = cnts[0]
        self.label = label
        self.blob_class = BlobClass.PERSON

        self.area = cv2.contourArea(cnts)
        self.perimeter = cv2.arcLength(cnts, True)
        M = cv2.moments(cnts)
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])

    def get_attributes(self):
        return [self.label, self.area, self.perimeter, self.blob_class]

class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2