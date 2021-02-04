from enum import Enum
import cv2

class Blob:

    def __init__(self, cnts):
        cnts = cnts[0]
        self.label = None
        self.blob_class = BlobClass.PERSON

        self.area = cv2.contourArea(cnts)
        self.perimeter = cv2.arcLength(cnts, True)
        M = cv2.moments(cnts)
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])


class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2