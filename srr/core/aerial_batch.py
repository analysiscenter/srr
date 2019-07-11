""" Batch for analysis of images and LiDAR data from UAVs """

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
        _ = dst
        path = '_'.join(self._make_path(ix, src).split('_')[:-1]) + '_mask.png'
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

    @action
    def _make_crops_(self, image, size):
        """Crop patches from original image and combine them into array.
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
        """Split crops from one image into separate images within new batch.
        """
        images = np.array([crop for img in self.images for crop in img] + [None])[:-1]
        index = DatasetIndex(np.arange(len(images)))
        batch = type(self)(index)
        batch.images = images

        masks = np.array([crop for img in self.masks for crop in img] + [None])[:-1]
        batch.masks = masks

        batch.orig_images = self.orig_images
        batch.orig_masks = self.orig_masks

        return batch
