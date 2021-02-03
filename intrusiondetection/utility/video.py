 # Video Editing Functions   

import cv2

def create_output_stream(cap, output_video_path):
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