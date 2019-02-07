""" Batch for analysis of images and LiDAR data from UAVs """

import numpy as np
import PIL

import batchflow as bf
from batchflow import ImagesBatch, action, inbatch_parallel
from batchflow.batch_image import transform_actions


@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
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
    
    @action
    def _make_crops_(self, image, size):
        """
        """
        crops = []
        imsize = image.size
        x_times = np.ceil(imsize[0] / size[0]).astype(int)
        y_times = np.ceil(imsize[1] / size[1]).astype(int)

        for y in range(y_times):
            for x in range(x_times):
                origin = (x*size[0], y*size[1])
                right_bottom = tuple(map(sum, zip(origin, size)))
                crops.append(image.crop((*origin, *right_bottom)))

        return np.array(crops + [None])[:-1]
    
    @action
    def unstack_crops(self):
        """
        """
        n_reps = [img.shape[0] for img in self.images]
        images = np.array([crop for img in self.images for crop in img] + [None])[:-1]
        index = bf.DatasetIndex(np.arange(len(images)))
        batch = self.__class__(index)
        batch.images = images
        
        masks = np.array([crop for img in self.masks for crop in img] + [None])[:-1]
        batch.masks = masks

        batch.orig_images = self.orig_images
        
        return batch
