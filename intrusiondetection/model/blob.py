from enum import Enum

class Blob:

    def __init__(self, label):
        self.label = label
        self.blob_class = BlobClass.PERSON 


class BlobClass(Enum):
    PERSON = 1
    OBJECT = 2