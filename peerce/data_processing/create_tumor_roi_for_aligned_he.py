import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

from pathlib import PosixPath

class TPath(PosixPath):
    # TrueStemPath
    @property
    def tstem(self):
        """The first path component, minus its suffixes."""
        name = self.name
        i = name.find('.')
        if 0 < i < len(name) - 1:
            return name[:i]
        else:
            return name
        
        
def find_pckl_match(path_wsi, meta_files):
    res = 'not_found'
    for f in meta_files:
        fstem = f.stem
        fstem = fstem.replace("_no_tumor_rois", "")
        if path_wsi.tstem == fstem:
            res = f
            break
    return res


def get_ometiff_tissue_mask(ome_tiff, staining_name='eosin', threshold = 'otsu', max_otsu_thresh=0.028, 
                            black_box_limit = (20, 20, 20)):
    
    staining_list = ['hematoxylin', 'eosin']

    img_rgb = ome_tiff.read_level(5)
    black_box_mask = np.all(img_rgb <= black_box_limit, axis=-1)
    img_rgb[black_box_mask] = 255
    # plt.imshow(img_rgb)
    # plt.show()
    if staining_name == 'grayscale':
        img_rgb = Image.fromarray(img_rgb)
        staining = img_rgb.convert('L')
        # create array, stuff numbers between 0 and 1 and invert (for correct threshold application later on)
        # divide by 100 to match eosin range more closely
        staining = (1 - np.array(staining)/255)/10
        staining[staining<0] = 0
    else:
        # Separate color stains (H&E) from the WSI image
        img_hed = rgb2hed(img_rgb)
        staining_index = staining_list.index(staining_name)
        staining = img_hed[:,:,staining_index]
        staining[staining<0] = 0
        # check if we want to calculate the otsu threshold

    if threshold == 'otsu':
        thresh = threshold_otsu(staining)
        print(f"Used Otsu's thresholding technique (threshold={thresh})")
        if thresh > max_otsu_thresh:
            print(f"Selected threshold too high, reduce from {thresh} to {max_otsu_thresh}")
            thresh = max_otsu_thresh
    else:
        thresh = threshold
        print(f"Used given threshold of {thresh}")

    # mask everything which is smaller than threshold
    tissue_mask = staining>thresh
    return tissue_mask