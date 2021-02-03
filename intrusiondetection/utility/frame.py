import numpy as np

# Frame Editing Functions

def binarize_mask(mask, threshold=None):
    ''' This method takes the mask and binarize it. The return is a mask of float64 with values 0 or 255
    '''
    res = np.zeros(mask.shape)
    if mask.dtype == bool:
        res[mask] = 255
    else:
        res[mask < threshold] = 255
    return res
