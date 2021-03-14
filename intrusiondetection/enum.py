from enum import Enum

class BlobClass(Enum):
    '''
        Enum to distinguish the type of the blob
    '''
    PERSON = 1
    OBJECT = 2
    FAKE = 3

    def __str__(self):
        return self.name

class BackgroundMethod(Enum):
    BLIND = 1
    SELECTIVE = 2