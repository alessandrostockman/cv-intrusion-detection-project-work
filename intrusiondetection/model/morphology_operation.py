import numpy as np
import cv2

class MorphologyOperation:

    def __init__(self, operation_type, kernel_size, kernel_shape=None):
        self.operation_type = operation_type
        if kernel_shape is None:
            self.kernel = np.ones(kernel_size, dtype=np.uint8)
        else:
            self.kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
