from enum import Enum

class BlobClass(Enum):
    '''
        Blobs possible classifications
    '''
    PERSON = 1
    OBJECT = 2
    FAKE = 3

    def __str__(self):
        return self.name

class BackgroundMethod(Enum):
    '''
        Background update methods
    '''
    BLIND = 1
    SELECTIVE = 2

class ParameterPreset(Enum):
    '''
        Parameters preset identifiers
    '''
    SLOW = 1
    MEDIUM = 2
    FAST = 3