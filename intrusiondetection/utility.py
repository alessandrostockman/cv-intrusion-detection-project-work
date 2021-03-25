import numpy as np
from matplotlib import pyplot as plt

from intrusiondetection.morphology import MorphOpsSet

# Distance Functions

def distance_manhattan(img1, img2):
    ''' 
        Performs manhattan distance on the given images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' 
        Performs euclidean distance on the given images
    '''
    return (np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)).astype(np.uint8)

# Plotting Functions

def subplot_images(cols, row_index=0, rows_number=1):
    cols_number = len(cols)
    plt.figure(figsize=(5 * cols_number, 2.5 * cols_number))
    
    for col_index, image_dict in enumerate(cols):
        plt.subplot(rows_number, cols_number, (col_index + 1) + row_index * cols_number)
        plt.axis('off')
        image_dict['object'].display(image_dict['key'], title=image_dict['title'], show=False, cols_number=cols_number)

def subplot_images_frames(frames, indexes, keys, title):
    if type(keys) is not list:
        keys = [keys]

    for row_index, key in enumerate(keys):
        subplot_images([
            {'object': fr, 'key': key, 'title': title + ' ' + str(idx)} 
            for fr, idx in zip(frames, indexes)
        ], row_index=row_index, rows_number=len(keys))

def subplot_morphology_steps(frame, morph_ops_list):
    frames = []
    operations = []
    
    for morph_op in morph_ops_list:
        operations.append(morph_op)
        new_frame = frame.copy()
        new_frame.apply_morphology_operators(MorphOpsSet(*operations))
        frames.append(new_frame)

    subplot_images_frames(frames, range(1, len(frames) + 1), 'mask_refined', "Morph Op")