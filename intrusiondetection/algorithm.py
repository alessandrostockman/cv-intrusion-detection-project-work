import cv2
import csv
import numpy as np

from intrusiondetection.utility.video import create_output_stream
import intrusiondetection.utility.parameters

from intrusiondetection.transformation import ChangeDetectionTransformation
from intrusiondetection.transformation import BinaryMorphologyTransformation
from intrusiondetection.transformation import ConnectedComponentTransformation

class IntrusionDetectionAlgorithm:

    def __init__(self, params):
        self.params = params
        self.cd = ChangeDetectionTransformation(params)
        self.bm = BinaryMorphologyTransformation(params)
        self.cc = ConnectedComponentTransformation(params)

    def execute(self):
        frame_index = 0
        cap = cv2.VideoCapture(self.params.input_video)
        out = create_output_stream(cap, self.params.output_video)
        out1 = create_output_stream(cap, str(self.params) + "_cd.avi")
        out2 = create_output_stream(cap, str(self.params) + "_bm.avi")
        out3 = create_output_stream(cap, str(self.params) + "_cc.avi")
        try:
            csv_file = open(self.params.output_text, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            while cap.isOpened():
                ret, frame = cap.read()
                self.params.frame = frame
                if not ret or frame is None:
                    print("Computation finished! Video saved in {}".format(self.params.output_video))
                    break

                frame1 = self.cd.apply(frame)
                frame2 = self.bm.apply(frame1)
                frame3 = self.cc.apply(frame2)

                #blobs, colored_frame = blob_detection(mask_output, frame, parameters)
                #append_text_output(idx, blobs, csv_writer)
                
                out.write(frame3)
                out1.write(np.tile(frame1.astype(np.uint8)[:,:,np.newaxis], 3) * 255)
                out2.write(np.tile(frame2[:,:,np.newaxis], 3) * 255)
                out3.write(self.params.adaptive_background.astype(np.uint8))
                frame_index += 1
        finally:
            csv_file.close()