""" Prepare DeepGlobe train masks """

# Renames mask from 'train' folder to the same name as image, but with
# '.png' extension. Also transforms mask from H*W*3 format (last axis has
# values 0 and 255, combinations of 0 and 255 encode classes) to H*W*1 format, where last axis
# has class number values from  0 to 6.

import os
import argparse
from multiprocessing import Pool
import numpy as np
from PIL import Image

def transform_mask_to_class_numbers(fname):
    """Transforms mask from 3-chanel binary encoding to flat mask.
    """
    mask = np.array(Image.open(fname))
    mask = mask // 128 # to 0 / 1 values
    # use 128 as suggested in https://arxiv.org/pdf/1806.03510.pdf
    base = 2 ** np.arange(3)[::-1] # binary base
    mask = (mask * base).sum(axis=-1).astype(np.uint8)
    Image.fromarray(mask).save(fname)

def main():
    """Apply mask-transforming function in parallel to all files in 'png' format
    in specified folder.
    """
    parser = argparse.ArgumentParser('')
    parser.add_argument('--path', dest='path', default=None,
                        help='Path to the files')
    args = parser.parse_args()
    path = args.path
    if not path:
        raise ValueError('You need to specify path to the files with `--path`.')

    names = os.listdir(path)
    mask_names = [os.path.join(path, name) for name in names if name.endswith('png')]

    pool = Pool(4)
    pool.map(transform_mask_to_class_numbers, mask_names)

    #close the pool and wait for the work to finish
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
