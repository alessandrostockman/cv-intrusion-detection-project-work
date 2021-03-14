import csv
import numpy as np
import cv2

from intrusiondetection.displayable import Frame, Background
from intrusiondetection.enum import BackgroundMethod

class Video:
    '''
    Class Video defining the video that will be written in output
    '''
    def __init__(self, input_video_path):
        self.frames = []
        self.backgrounds = []
        self.__frame_index = 0
        self.__cap = cv2.VideoCapture(input_video_path)

        self.__w = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__h = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fps = self.__cap.get(cv2.CAP_PROP_FPS)

        self.load_video()
        
    def load_video(self):
        '''
            Loads a list of frames for the current video
        '''
        while self.__cap.isOpened():
            ret, frame_image = self.__cap.read()
            if not ret or frame_image is None:
                break

            self.frames.append(Frame(frame_image))

    def process_backgrounds(self, update_mode, initial_background, alpha, threshold=None, distance=None, morph_ops=None):
        """
            Method used only for demonstration purposes, returns a list of dynamic backgrounds for a given video either in BLIND or SELECTIVE update mode  
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

    def intrusion_detection(self, params, initial_background, tuning=False):
        '''
            Application of the intrusion detection algorithm
        '''

        outputs_base_name = params.output_base_name + "_" if tuning else params.output_directory + "/"
        self.__outputs = {output_type: {
            key: self.create_output_stream(self.__w, self.__h, self.__fps, outputs_base_name + output_type + "_" + key + ".avi") for key in outputs
        } for output_type, outputs in params.output_streams.items()}

        try:
            csv_file = open(outputs_base_name + "text.csv", mode='w')
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

                for output_type, outputs in self.__outputs.items():
                    obj = fr if output_type == 'foreground' else prev_bg
                    for key, out in outputs.items():
                        output_image = getattr(obj, key, None)

                        if output_image is not None:
                            if output_image.shape[-1] != 3:
                                output_image = np.tile(output_image[:,:,np.newaxis], 3)
                            if output_image.dtype != np.uint8:
                                output_image = output_image.astype(np.uint8)                        
                            out.write(output_image)

                fr.generate_text_output(csv_writer, frame_index=self.__frame_index)
                self.__frame_index += 1
                prev_fr = fr
                prev_bg = bg
        finally:
            csv_file.close()
    
    def create_output_stream(self, w, h, fps, output_video_path):
        '''
            Creates a video writer reference for output_video_path
        '''
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w,  h))

        return out
