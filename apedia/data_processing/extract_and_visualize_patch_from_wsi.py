import numpy as np
from matplotlib import pyplot as plt
import random
import collections.abc


def show(mask):
    plt.imshow(mask)
    plt.show()

##########################################################
# Start of cell 9520a591
##########################################################
def get_wsi_patch(reduced_mask_coordinates, borders, wsi, size=(512, 512), level=1):
    """
    Return patch based on coordinates from the reduced mask. 
    Patch as a default size of (512, 512)
    """
    height = (reduced_mask_coordinates[0] + borders['mask_up']) * borders['scale']
    width = (reduced_mask_coordinates[1] + borders['mask_left']) * borders['scale']
    img = wsi.read_region((width, height), level, size)
    img = img.convert('RGB')
    arr = np.array(img)
    return arr



def get_wsi_patch_ome(reduced_mask_coordinates, borders=None, wsi=None, size=(512, 512), level=1, scale=2**5):
    """
    Return patch based on coordinates from the reduced mask. 
    Patch as a default size of (512, 512)
    """
    if borders is not None:
        scale = borders['scale']
    height = reduced_mask_coordinates[0] * scale
    width = reduced_mask_coordinates[1] * scale
    arr = wsi.read_region((height, width), level=1, size=512)
    return arr


def get_rand_mask_location(mask, borders, padding=32):
    '''
    Return a random mask location
    '''
    height = borders['mask_down'] - borders['mask_up']
    width = borders['mask_right'] - borders['mask_left']
    rand_height = np.random.randint(0, height-padding)
    rand_width = np.random.randint(0, width-padding)
    return rand_height, rand_width


def get_reduced_mask(mask, borders):
    '''
    Reduce returned mask to borders confining the tissue the slide has.
    '''
    return mask[borders['mask_up']:borders['mask_down'],borders['mask_left']:borders['mask_right']]


def get_mask_patch(mask, loc, size=32):
    '''
    Return patch from mask. Array slicing is enough, as mask is array and not WSI.
    '''
    return mask[loc[0]:loc[0]+size, loc[1]:loc[1]+size]


def get_tumor_mask_reduced_mask_coordinates(reduced_mask_coordinates, borders, mask=None, size=(512, 512), scale=2**5):
    """
    Return tumor_mask based on coordinates from the reduced mask. 
    Patch as a default size of (512, 512)
    """
    if borders is not None:
        scale = borders['scale']
    # need to divide borders['scale'] by 2, as the mask is at level 1, while the wsi patched using the location at level 0, 
    # regardless of the level used for the patch
    height = int(reduced_mask_coordinates[0] * scale / 2)
    width = int(reduced_mask_coordinates[1] * scale / 2)
    arr = mask[height:height+size[0], width:width+size[1]]
    return arr


def get_tissue_coverage(patch, granularity=16, sensitivity=1):
    '''
    This returns the coverage of a patch. This menas, how many parts are covered in tissue.
    'granularity' sets how many patches are tested (e.g. 16 means the patch is sliced into 4x4 patches which are 
    individually tested). Default is at 16.
    'sensitivity' determines how many tissue pixels need to be found in each patch. Default is at 1
    '''
    check_detail = int(granularity**0.5)
    coverage = []
    step = int(patch.shape[0]/check_detail)
    for i in range(check_detail):
        for j in range(check_detail):
            coverage.append(patch[i*step:i*step+step, j*step:j*step+step].sum()>sensitivity)
    return np.mean(coverage)


def get_rand_tissue_patch(wsi, mask, borders, coverage=0, sensitivity=1, viz=True):
    '''
    Return tissue patch based on wsi, it's tissue mask and pre-defined borders.
    The tissue patch is within the pre-determined borders.
    Given the coverage & sensitivity values, the function randomly looks for patches 
    until it fulfills the minimum coverage based on the set sensitivity value.
    Found patch is returned.
    '''
    loc = get_rand_mask_location(mask, borders)
    reduced_mask = get_reduced_mask(mask, borders)
    mask_patch = get_mask_patch(reduced_mask, loc)
    tissue_coverage = get_tissue_coverage(mask_patch, sensitivity=sensitivity)
    while tissue_coverage<=coverage:
        # print(f"not enough coverage ({tissue_coverage})")
        loc = get_rand_mask_location(mask, borders)
        patch = get_mask_patch(reduced_mask, loc)
        tissue_coverage = get_tissue_coverage(patch, sensitivity=sensitivity)
    else:
        print(f"Display patch from {borders['wsi_path'].stem}")
        print(f"coverage: {tissue_coverage}")
        # print(loc)
        tissue_patch = get_wsi_patch(loc, borders, wsi)
        if viz:
            show(tissue_patch)
    return tissue_patch   


def find_all_tissue_patch_locations(reduced_mask, step_size=32, sensitivity=1, coverage=0):
    '''
    Return a list of tuples containing all patch locations meeting the set conditions of sensitivity and coverage.
    Tuple consists of (height, width, tissue_coverage).
    Assumes that step size is equivalent to patch size.
    '''
    patch_locations = []
    steps_height, steps_width = np.divide(reduced_mask.shape, step_size).astype(int)
    redux_mask_height, redux_mask_width = reduced_mask.shape
    for step_height in range(steps_height):
        height = step_height*step_size
        if height > (redux_mask_height+step_size):
            continue
        for step_width in range(steps_width):
            width = step_width*step_size
            if width > (redux_mask_width+step_size):
                continue

            mask_patch = get_mask_patch(reduced_mask, (height, width), size=step_size)
            tissue_coverage = get_tissue_coverage(mask_patch, sensitivity=sensitivity)
            if tissue_coverage > coverage:
                patch_locations.append((height, width, tissue_coverage))
    return patch_locations
##########################################################
# End of cell 9520a591
##########################################################

##########################################################
# Start of cell 60e98662
##########################################################
def get_tissue_thumbnail(wsi, borders, level=7, reduce_to_max=False, thumbnail_max=512, viz=False):
    mask_height = borders['wsi_down']-borders['wsi_up']
    mask_width = borders['wsi_right']-borders['wsi_left']

    # select tissue region with size of mask from certain level
    
    # weird bug within openslide. Read region behaves unpredictably, if x & y coordinates don't point to 
    # a multiple of 2**level
    x_coord = int(np.round(borders['wsi_left']/2**level))*2**level
    y_coord = int(np.round(borders['wsi_up']/2**level))*2**level
    
    tissue = wsi.read_region((x_coord, y_coord), level, (mask_width//2**level, mask_height//2**level))
    if reduce_to_max:
        # reduce thumbnail size to max 512 edge length
        smaller_size = np.round(np.multiply(tissue.size, thumbnail_max/max(tissue.size))).astype(int)
        tissue = tissue.resize(smaller_size)
    # convert to RGB & array
    tissue = np.array(tissue.convert('RGB'))
    if viz:
        show(tissue)
    return tissue


def get_percentage_patch_location(loc, borders, print_info=True):
    mask_size = [borders['mask_down']-borders['mask_up'], borders['mask_right']-borders['mask_left']]
    patch_perc = np.divide(loc, mask_size)
    if print_info:
        print(f"Patch from height at {patch_perc[0]*100:.2f}% & width at {patch_perc[1]*100:.2f}%")
    return patch_perc


def blacken_thumbnail(thumbnail, patch_perc, sz=16):
    thumb_patch_location = np.multiply(thumbnail.shape[:-1], patch_perc).astype(int)
    # thumbnail[thumb_patch_location[0]-(np.floor(sz/2).astype(int)-8):thumb_patch_location[0]+(np.ceil(sz/2).astype(int)+8),
    #           thumb_patch_location[1]-(np.floor(sz/2).astype(int)-8):thumb_patch_location[1]+(np.ceil(sz/2).astype(int)+8)] = (255, 0, 0) # previously: =0
    # thumbnail[thumb_patch_location[0]-(np.floor(sz/2).astype(int)):thumb_patch_location[0]+(np.ceil(sz/2).astype(int)),
    #         thumb_patch_location[1]-(np.floor(sz/2).astype(int)):thumb_patch_location[1]+(np.ceil(sz/2).astype(int))] = (255, 0, 0) # previously: =0
    thumbnail[thumb_patch_location[0]:thumb_patch_location[0]+sz,
            thumb_patch_location[1]:thumb_patch_location[1]+sz] = (255, 0, 0) # previously: =0
    return thumbnail

def get_location_macro_viz(location, wsi, borders, ome=False):
    '''
    Visualize from a macro scale where the patch is taken from.
    '''
    if ome:
        thumb = wsi.get_thumbnail(512, make_array=True)
    else:
        thumb = get_tissue_thumbnail(wsi, borders)
    
    # if location is an empty list, just return the thumbnail :)
    if len(location) == 0:
        return thumb
        
    # check if multiple locations exist (in that case, the first element of the list is a tuple)
    # if not multiple locations - encapsule loc in []
    if not isinstance(location[0], (collections.abc.Sequence, np.ndarray)):
        location = [location]
    
    if borders is None:
        # equivalent to shape of level 5
        wsi_dimensions = wsi.level_dimensions()[5]
        borders = {
            'mask_up': 0,
            'mask_down': wsi_dimensions[0],
            'mask_left': 0,
            'mask_right': wsi_dimensions[1]
        }
    
    size_decrease_thumb = thumb.shape[0]/wsi.level_dimensions()[1][0]
    thumb_patch_size = max(int(np.round(512*size_decrease_thumb)), 1)
    for loc in location:
        patch_perc = get_percentage_patch_location(loc[:2], borders, print_info=False)
        thumb = blacken_thumbnail(thumb, patch_perc, sz=thumb_patch_size)
    return thumb

def viz_patch_and_location(location, wsi, borders, ome=False):
    thumb = get_location_macro_viz(location, wsi, borders, ome)

    
    # check if multiple locations exist (in that case, the first element of the list is a tuple)
    # if not multiple locations - encapsule loc in []
    all_location_figure=True
    # import pdb; pdb.set_trace()
    if not isinstance(location[0], (collections.abc.Sequence, np.ndarray)):
        location = [location]
        all_location_figure=False
    
    
    
    patch_loc = random.choice(location)
    if ome:
        patch = get_wsi_patch_ome(patch_loc, borders, wsi)
    else:
        patch = get_wsi_patch(patch_loc, borders, wsi)

    if all_location_figure:
        patch_tumb = get_location_macro_viz(patch_loc, wsi, borders)
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        ax[0].imshow(patch)
        ax[1].imshow(patch_tumb)
        ax[2].imshow(thumb)
        ax[2].set_title(f'All patch locations ({len(location)})')
    else:   
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(patch)
        ax[1].imshow(thumb)
    ax[0].set_title('Randomly chosen patch')
    ax[1].set_title('Location of patch')
    plt.tight_layout()
    plt.show()
##########################################################
# End of cell 60e98662
##########################################################
