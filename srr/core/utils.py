""" Utility functions for SRR """

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from ..batchflow.batchflow.models.tf.losses import dice, softmax_cross_entropy


def get_origins(mask, classes, crop_shape=(128, 128), proba=0.5, seed=None):
    """Calculates coordinates of top left corner of cropping box.

    With probability `proba` finds all pixels with values from `classes`, samples one
    of them and returns top left corner coordinates of the crop with center at that point.
    With probability 1 - `proba` samples top left corner coordinates of cropping box
    randomly.
    The resulting cropping box is guaranteed to lie within the original image.

    Parameters
    ----------
    mask : PIL.Image.Image
        Grayscale mask with integer values that represent corresponding classes.
    classes : tuple of ints
        Classes that are used in sampling procedure.
    crop_shape : tuple of ints
        Needed to guarantee that resulting cropping box lie within the original image.
    proba : float
        Probability of sampling cropping box around a pixel with value from `classes`.
    seed : int or 1-d array_like
        Argument for `np.random.seed`.
    """
    background_shape = mask.size
    np.random.seed(seed)

    classes_in_mask = [val for _, val in mask.getcolors()]
    if np.random.uniform() <= proba and np.any(np.isin(classes_in_mask, classes)):
        arr_mask = np.array(mask)
        good_points = np.where(np.isin(arr_mask, classes))
        center_index = np.random.randint(0, len(good_points[0]))
        origin = [good_points[0][center_index] - int(np.ceil(crop_shape[0] / 2)),
                  good_points[1][center_index] - int(np.ceil(crop_shape[1] / 2))]
        origin[0] = min(max(0, origin[0]), background_shape[0] - int(np.floor(crop_shape[0] / 2)))
        origin[1] = min(max(0, origin[1]), background_shape[1] - int(np.floor(crop_shape[1] / 2)))
    else:
        origin = (np.random.randint(background_shape[0] - crop_shape[0] + 1),
                  np.random.randint(background_shape[1] - crop_shape[1] + 1))
    return origin[::-1]

def make_mask(mask, classes):
    """
    Prepare masks for the model.

    Parameters
    ----------
    mask : ndarray
        (x, y) or (x, y, 1) array of ints, representing pixel classes.
    classes: tuple of integers > 0
        Classes from source mask to be included in resulting mask. Zero is reserved for background.

    Returns
    -------
    new_mask : ndarray
        (x, y, len(classes)) array with one-hot mask.
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask, -1)
    new_mask = np.zeros((*mask.shape, len(classes) + 1))

    for i, label in enumerate(classes):
        new_mask[:, :, i+1] = mask == label
    new_mask[:, :, 0] = np.sum(new_mask, axis=-1) == 0
    return new_mask.astype(np.uint8)

def stack_crops(crops, n_hor, n_ver):
    """Gathers an image from n_hor * n_ver ndarrays.

    Performs `n_ver` horizontal stacks of `n_hor` crops, then stacks all of them vertically.
    """
    return np.vstack([np.hstack(crops[i*n_hor:(i+1)*n_hor]) for i in range(n_ver)])

def gather_img_mask_pred(batch, original_size, crop_shape):
    """ Gathers images from crops of `images`, `masks` and `predictions` components of 
    `batch`.

    Notes
    -----
    Works only with batches of size 1. All three components should be present in batch.
    """
    n_hor, n_ver = np.ceil(np.array(original_size) / crop_shape).astype(int)
    img = stack_crops(batch.images, n_hor, n_ver)[:original_size[0], :original_size[1]]
    mask = stack_crops(batch.masks, n_hor, n_ver)[:original_size[0], :original_size[1]]
    pred = stack_crops(batch.predictions, n_hor, n_ver)[:original_size[0], :original_size[1]]

    return img, mask, pred

def ce_dice_loss(labels, logits, alpha=0.75, *args, **kwargs):
    """Weighted sum of CE and DICE losses.
    """
    _ = args, kwargs
    ce_loss = alpha * softmax_cross_entropy(labels=labels, logits=logits, loss_collection=None)
    dice_loss = (1-alpha) * dice(labels, logits, loss_collection=None)
    loss = ce_loss + dice_loss
    tf.losses.add_loss(loss)

    return loss

def plot_img_pred_mask(img, pred, mask, print_forest_shares=False, figsize=(30, 40)):
    """Displays image, prediction and mask.

    If print_forest_shares is True, prints shares of forest on pred and on mask.
    Share of forest on pred is calculated with .5 threshold.
    """

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax1.imshow(img)
    ax1.set_title('Image', size=16)
    ax2.imshow(pred, vmin=0, vmax=1)
    ax2.set_title('Prediction', size=16)
    ax3.imshow(mask, vmin=0, vmax=1)
    ax3.set_title('Mask', size=16)
    plt.show()
    if print_forest_shares:
        print("Percentage of forest cover: predicted - {:.2f}%, ground truth - {:.2f}%"
              "".format((np.sum(pred > 0.5) / pred.size) * 100, np.sum(mask) / mask.size * 100))
