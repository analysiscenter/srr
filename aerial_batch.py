""" Batch for analysis of images and LiDAR data from UAVs """

import numpy as np
import PIL

import batchflow as bf
from batchflow import ImagesBatch, action, inbatch_parallel

def get_origs(mask, crop_shape=(128,128), p=0.5, seed=None):
    """
    """
    background_shape = mask.size
    np.random.seed(seed)
    arr_mask = np.array(mask)
    if np.random.uniform()>=p and np.any(arr_mask):
        good_points = np.where(arr_mask>0)
        center_index = np.random.randint(0, len(good_points[0]))
        origin = [good_points[0][center_index]-int(np.ceil(crop_shape[0]/2)), 
                  good_points[1][center_index]-int(np.ceil(crop_shape[1]/2))]
        origin[0] = min(max(0, origin[0]), background_shape[0] - int(np.floor(crop_shape[0] / 2)))
        origin[1] = min(max(0, origin[1]), background_shape[1] - int(np.floor(crop_shape[1] / 2)))
    else:
        origin = (np.random.randint(background_shape[0]-crop_shape[0]+1),
                  np.random.randint(background_shape[1]-crop_shape[1]+1))
    return origin[::-1]

def make_mask(mask, classes=(1, 2)):
    """
    Notes
    -----
    
    Label | Class
    *************
    0     | Unknown
    1     | Water
    2     | Forest land
    3     | Urban land
    5     | Rangeland
    6     | Agriculture land
    7     | Barren land
    """
    
    mask = np.squeeze(mask, -1)
    new_mask = np.zeros((*mask.shape, len(classes)) + 1)
    
    for i, class in enumerate(classes):
        new_mask[:, :, i+1] = mask == class
    new_mask[:, :, 0] = np.sum(new_mask, axis=-1) == 0
    return new_mask.astype(np.uint8)

class AerialBatch(ImagesBatch):
    """
    """

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_mask(self, ix, src=None, dst='masks'):
        """
        """
        path = self._make_path(ix, src)[:-7] + 'mask.png'
        return PIL.Image.open(path)

    @action
    def load(self, *args, src=None, fmt=None, components=None, **kwargs):
        """ Load data.

        .. note:: if `fmt='images' or 'png'` than ``components`` must be a single component (str).
        .. note:: All parameters must be named only.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to download.
        components : str, sequence
            components to download.
        """
        if fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=components)
        elif fmt == 'mask':
        	return self._load_mask(src, dst=components)
        return super().load(src=src, fmt=fmt, components=components, *args, **kwargs)