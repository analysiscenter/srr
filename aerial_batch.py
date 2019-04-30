""" Batch for analysis of images and LiDAR data from UAVs """

import numpy as np
import PIL

from batchflow import ImagesBatch, action, inbatch_parallel

def get_origs(mask, crop_shape=(128, 128), p=0.5, seed=None):
    """Function that find non-zero mask points, ramdomly choses one
    and returns coordinated of the crop with center at that point.
    """
    background_shape = mask.size
    np.random.seed(seed)
    arr_mask = np.array(mask)
    if np.random.uniform() >= p and np.any(arr_mask):
        good_points = np.where(arr_mask > 0)
        center_index = np.random.randint(0, len(good_points[0]))
        origin = [good_points[0][center_index]-int(np.ceil(crop_shape[0]/2)),
                  good_points[1][center_index]-int(np.ceil(crop_shape[1]/2))]
        origin[0] = min(max(0, origin[0]), background_shape[0] - int(np.floor(crop_shape[0] / 2)))
        origin[1] = min(max(0, origin[1]), background_shape[1] - int(np.floor(crop_shape[1] / 2)))
    else:
        origin = (np.random.randint(background_shape[0]-crop_shape[0]+1),
                  np.random.randint(background_shape[1]-crop_shape[1]+1))
    return origin[::-1]

class AerialBatch(ImagesBatch):
    """Class for reading and processing of aerial images.
    """

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_mask(self, ix, src=None, dst='masks'):
        """Loads masks

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            Path to the folder with an image. Mask path is inferred from
            image name. E.g. mask name '121_maks.png' is supposed to be
            a mask for '121_sat.png' image. Mask should in png format.
        dst : str
            Component to write images to.
        fmt : str
            Format of an image.
        """
        path = self._make_path(ix, src).split('_')[0] + '_mask.png'
        return PIL.Image.open(path)

    @action
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data.

        .. note:: if `fmt='images' or 'png'` than ``components`` must be a single component (str).
        .. note:: All parameters must be named only.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather', 'mask'}
            Format of the file to download.
        components : str, sequence
            components to download.
        """
        if fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=dst)
        if fmt == 'mask':
            return self._load_mask(src, dst=dst)
        return super().load(src=src, fmt=fmt, dst=dst, *args, **kwargs)
