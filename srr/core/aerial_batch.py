""" Batch for analysis of images from UAVs """

import numpy as np
import PIL

from ..batchflow import ImagesBatch, DatasetIndex, action, inbatch_parallel
from ..batchflow.batchflow.batch_image import transform_actions


@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class AerialBatch(ImagesBatch):
    """Class for reading and processing of aerial images.
    """

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_mask(self, ix, src=None, dst='masks'):
        """Load masks.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            Path to the folder with an image.
        dst : str
            Component to write images to.
        """
        _ = dst
        path = '_'.join(self._make_path(ix, src).split('_')[:-1]) + '_mask.png'
        return PIL.Image.open(path)

    @action
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather', 'mask'}
            Format of the file to download.
        components : str, sequence
            components to download.
            
        Notes
        -----
        Mask path is inferred from image name. E.g. mask name '121_mask.png' is supposed to be
        a mask for '121_sat.png' image. Masks should be in png format.
        """
        if fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=dst)
        if fmt == 'mask':
            return self._load_mask(src, dst=dst)
        return super().load(src=src, fmt=fmt, dst=dst, *args, **kwargs)

    @action
    def _make_crops_(self, image, shape):
        """Crop patches from original image and combine them into array.

        This action should be applied to instances of `PIL.Image.Image` class.

        Parameters
        ----------
        shape : tuple of ints
            shape of the resulting crops.
        """
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise ValueError("shape should be a tuple of size 2")
        crops = []
        imsize = image.size
        x_times = np.ceil(imsize[0] / shape[0]).astype(int)
        y_times = np.ceil(imsize[1] / shape[1]).astype(int)

        for y in range(y_times):
            for x in range(x_times):
                left_top = (x*size[0], y*size[1])
                right_bottom = (left_top[0] + shape[0], left_top[1] + shape[1])
                crops.append(image.crop((*top_left, *right_bottom)))

        return np.array(crops, dtype=object)

    @action
    def unstack_crops(self):
        """Split crops from one image into separate images within new batch.

        Note
        ----
        This action rebuilds index and keeps only `images` and `masks` components.
        """
        images = np.array([crop for img in self.images for crop in img], dtype=object)
        index = DatasetIndex(np.arange(len(images)))
        batch = type(self)(index)
        batch.images = images
        
        if 'masks' in self.components:
            masks = np.array([crop for img in self.masks for crop in img], dtype=object)
            batch.masks = masks

        return batch
