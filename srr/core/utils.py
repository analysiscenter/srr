""" Utility functions for SRR """

import numpy as np
import tensorflow as tf
from ..batchflow.batchflow.models.tf.losses import dice, softmax_cross_entropy


def get_origs(mask, classes, crop_shape=(128, 128), p=0.5, seed=None):
    """Function that find non-zero mask points, ramdomly choses one
    and returns coordinated of the crop with center at that point.
    """
    background_shape = mask.size
    np.random.seed(seed)
    arr_mask = np.array(mask)
    classes_in_mask = np.isin(arr_mask, classes)
    if np.random.uniform() <= p and np.any(classes_in_mask):
        good_points = np.where(classes_in_mask)
        center_index = np.random.randint(0, len(good_points[0]))
        origin = [good_points[0][center_index]-int(np.ceil(crop_shape[0]/2)),
                  good_points[1][center_index]-int(np.ceil(crop_shape[1]/2))]
        origin[0] = min(max(0, origin[0]), background_shape[0] - int(np.floor(crop_shape[0] / 2)))
        origin[1] = min(max(0, origin[1]), background_shape[1] - int(np.floor(crop_shape[1] / 2)))
    else:
        origin = (np.random.randint(background_shape[0]-crop_shape[0]+1),
                  np.random.randint(background_shape[1]-crop_shape[1]+1))
    return origin[::-1]

def make_mask(mask, classes):
    """
    Prepare masks for the model.

    Parameters
    ----------
    mask : ndarray
        (x, y) or (x, y, 1) array with inters representing pixel classes.
    classes: tuple of integers > 0
        Classes to be included in resulting mask. Zero is reserved for background.

    Returns
    -------
    new_mask : ndarray
        (x, y, len(classes)) array with one-hot mask.

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
    new_mask = np.zeros((*mask.shape, len(classes) + 1))

    for i, label in enumerate(classes):
        new_mask[:, :, i+1] = mask == label
    new_mask[:, :, 0] = np.sum(new_mask, axis=-1) == 0
    return new_mask.astype(np.uint8)

def gather_image(batch, preds, crop_shape):
    """ Gathers image from crops

    Notes
    -----
    Needs refactoring
    """
    size = np.ceil(np.array(batch.orig_images[0].size) / crop_shape).astype(int)
    img = np.vstack([np.hstack(batch.images[i*size[0]:(i+1)*size[0]]) for i in range(size[1])])
    mask = np.vstack([np.hstack(batch.masks[i*size[0]:(i+1)*size[0]]) for i in range(size[1])])
    pred = np.vstack([np.hstack(preds[i*size[0]:(i+1)*size[0]]) for i in range(size[1])])

    return img, mask, pred

def ce_dice_loss(labels, logits, alpha=0.75, *args, **kwargs):
    """Weighted sum of BCE and DICE losses.
    """
    _ = args, kwargs
    ce_loss = alpha * softmax_cross_entropy(labels=labels, logits=logits, loss_collection=None)
    dice_loss = (1-alpha) * dice(labels, logits, loss_collection=None)
    loss = ce_loss + dice_loss
    tf.losses.add_loss(loss)

    return loss
