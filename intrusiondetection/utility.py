import numpy as np
from matplotlib import pyplot as plt
import cv2

# Distance Functions

def distance_manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    #TODO Which version to use?
    return (np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)).astype(np.uint8)
    
    img1 = np.tile(img1.astype(float)[:,:,np.newaxis], 3)
    img2 = np.tile(img2.astype(float)[:,:,np.newaxis], 3)
    return np.sqrt(np.sum((img2 - img1) ** 2, axis=-1))

def subplot_images(image_dicts):
    plt.figure(figsize=(20, 10))
    for index, image_dict in enumerate(image_dicts):
        #img = getattr(image_dict['object'], image_dict['key'])
        plt.subplot(1, len(image_dicts), index+1)
        plt.axis('off')
        if image_dict['title'] is None:
             image_dict['title'] = None
        image_dict['object'].display(image_dict['key'], title=image_dict['title'], show=False)
        #plt.imshow(img, cmap='gray', vmin=0, vmax=255)


def hue_to_rgb(hue):
    return tuple(cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0,0])
