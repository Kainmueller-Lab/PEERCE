# adapted from https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L16
from tifffile import TiffFile
import numpy as np
from PIL import Image


def get_crop(page, attr_page, i0, j0, h, w):
    # for ometiff, only the first color channel page has attributes such as imagewidth et al.
    # for this, we always use the first page for attributes (-> attr_page )
    """Extract a crop from a TIFF image file directory (IFD).
    
    Only the tiles englobing the crop area are loaded and not the whole attr_page.
    This is usefull for large Whole slide images that can't fit int RAM.
    Parameters
    ----------
    attr_page : Tiffattr_page
        TIFF image file directory (IFD) from which the crop must be extracted.
    i0, j0: int
        Coordinates of the top left corner of the desired crop.
    h: int
        Desired crop height.
    w: int
        Desired crop width.
    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.
    """

    if not attr_page.is_tiled:
        raise ValueError("Input attr_page must be tiled.")

    im_width = attr_page.imagewidth
    im_height = attr_page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
        raise ValueError("Requested crop area is out of image bounds.")

    tile_width, tile_height = attr_page.tilewidth, attr_page.tilelength
    i1, j1 = i0 + h, j0 + w

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty((attr_page.imagedepth,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    attr_page.samplesperpixel), dtype=attr_page.dtype)

    fh = page.parent.filehandle

    jpegtables = attr_page.tags.get('JPEGTables', None)
    if jpegtables is not None:
        jpegtables = jpegtables.value

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            # tile, indices, shape = attr_page.decode(data, index, jpegtables)
            # import pdb; pdb.set_trace()
            tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]


class OmeTiffFile:
    '''
    Always returns arrays with channels in the last dimension
    '''
    
    def __init__(self, *args, **kwargs):
            self._tiff_file = TiffFile(*args, **kwargs)

    def level_count(self):
        return len(self._tiff_file.series[0].levels)

    def level_dimensions(self):
        level_dims = []
        for i in range(self.level_count()):
            dimensions = self._tiff_file.series[0].levels[i].shape[1:]
            level_dims.append(dimensions)
        return level_dims

    def get_thumbnail(self, size, make_quadratic=False, make_array=False):
        level_dims = self.level_dimensions()
        for i in range(len(level_dims)-1, -1, -1):
            if max(level_dims[i]) > size:
                break
        
        thumbnail = self._tiff_file.series[0].levels[i].asarray()
        thumbnail = thumbnail.transpose((1,2,0))
        thumbnail = Image.fromarray(thumbnail)

        resize_mult = size / max(thumbnail.size)
        target_size = np.multiply(thumbnail.size, resize_mult).round().astype(int)
        thumbnail = thumbnail.resize(target_size)
        
        if make_quadratic:
            x, y = target_size
            new_im = Image.new('RGB', (size, size), (255, 255, 255))
            new_im.paste(thumbnail, (int((size - x) / 2), int((size - y) / 2)))
            thumbnail = new_im
        
        if make_array:
            thumbnail = np.array(thumbnail)
        
        return thumbnail
                
    def read_region(self, location=(50000, 30000), size=512, level=0, make_image=False):
        height = location[0]
        width = location[1]
        scale = 0.5**level
        
        adjusted_height = int(height*scale)
        adjusted_width = int(width*scale)
        
        patch = []
        attr_page = self._tiff_file.series[0].levels[level].pages[0]
        for page in self._tiff_file.series[0].levels[level].pages:
            patch_channel = get_crop(page, attr_page, adjusted_height, adjusted_width, size, size).squeeze()
            patch.append(patch_channel)
        
        patch = np.stack(patch, axis=-1)
        
        if make_image:
            patch = Image.fromarray(patch)
        
        return patch

    def read_level(self, level):
        level = self._tiff_file.series[0].levels[level].asarray()
        level = level.transpose([1,2,0])
        return level

    def close(self):
        self._tiff_file.close()