from enum import Enum
import cv2

class Blob:
    def __init__(self, cnts, frame):
        cnts = cnts[0]
        self.cnts = cnts
        self.label = None
        self.blob_class = BlobClass.PERSON

        self.area = cv2.contourArea(cnts)
        self.perimeter = cv2.arcLength(cnts, True)
        M = cv2.moments(cnts)
        self.cx = int(M['m10']/(M['m00']))
        self.cy = int(M['m01']/(M['m00']))

        sum = len(self.cnts)
        val = 0
        frame = frame[:,:,0]
        x_max, y_max = frame.shape
        for coord in self.cnts:
            y, x = coord[0][0], coord[0][1]

            if y <= 0 or y >= y_max - 1 or x <= 0 or x >= x_max - 1:
                dx = 0
                dy = 0
            else:
                dx = self.i4y(frame, x, y+1) - self.i4y(frame, x, y-1)
                dy = self.i4x(frame, x+1, y) - self.i4x(frame, x-1, y)

            val += max(abs(dx), abs(dy))
        self.edge_strength = val / sum

    def is_true_object(self, threshold):
        return self.edge_strength > threshold

    def i4x(self, frame, i, j):
        return 1/4 * frame[i, j-1] + 2 * frame[i, j] + frame[i, j+1]

    def i4y(self, frame, i, j):
        return 1/4 * frame[i-1, j] + 2 * frame[i, j] + frame[i+1, j]


class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2