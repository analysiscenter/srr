""" Batch for analysis of images and LiDAR data from UAVs """

import numpy as np
import PIL

import batchflow as bf
from batchflow import ImagesBatch, action, inbatch_parallel

def get_origs(mask, crop_shape=(128,128), seed=None):
    background_shape = mask.size
    np.random.seed(seed)

    origin = (np.random.randint(background_shape[0]-crop_shape[0]+1),
              np.random.randint(background_shape[1]-crop_shape[1]+1))
    return origin[::-1]

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