import numpy as np
import cv2

class MorphOp:

    def __init__(self, operation_type, kernel_size, kernel_shape=None):
        self.operation_type = operation_type
        self.kernel_size = kernel_size
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
        self.ops = iter(ops)

    def __iter__(self):
        return self.ops

    def __str__(self):
        return "".join(str(x) for x in self.ops)