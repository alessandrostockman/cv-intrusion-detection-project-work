from enum import Enum
import cv2

class Blob:

    def __init__(self, label, cnts):
        cnts = cnts[0]
        self.label = label
        self.blob_class = BlobClass.PERSON

        self.area = cv2.contourArea(cnts)
        M = cv2.moments(cnts)
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])
        self.perimeter = cv2.arcLength(cnts, True)


class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2