"""
@brief   Script that computes iteratively the mean and unbiased sample 
         standard deviation.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    15 Dec 2021.
"""

import argparse
import numpy as np
import os
import copy
import tqdm
#import random
#import pandas as pd
#import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
#import sklearn.manifold
#import multiprocessing
#import colour

# My imports
import torchseg.data_loader
import torchseg.utils


# Get dictionary from class index to class string, e.g. 0 -> u'Attached gingiva'
idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the ODSI-DB dataset (required: True)',
        '-m': 'Path to the output mean (required: True)',
        '-s': 'Path to the output std (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='PyTorch segmenter.')
    parser.add_argument('-i', '--input', required=True, type=str, 
        help=help('-i'))
    parser.add_argument('-m', '--mean', required=True, type=str, 
        help=help('-m'))
    parser.add_argument('-s', '--std', required=True, type=str, 
        help=help('-s'))

    # Read parameters
    args = parser.parse_args()
    
    return args


def validate_cmdline_params(args):
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')

    if os.path.exists(args.mean):
        raise RuntimeError('[ERROR] Mean file already exists.')

    if os.path.exists(args.std):
        raise RuntimeError('[ERROR] Std file already exists.')

    return args


def read_files(path: str):
    """
    @param[in]  path  Path to the folder with the ODSI-DB data.
    @returns a tuple (imgs, segs) containing two lists of paths to images and 
             segmentation labels.
    """
    # Get list of TIFF files
    tiffs = [f for f in torchseg.utils.listdir(path) if '.tif' in f]

    # Get list of segmentation files
    segs = [os.path.join(path, f) for f in tiffs if f.endswith('_masks.tif')]

    # Get list of image files (we ignore images without segmentation)
    imgs = [f.replace('_masks.tif', '.tif') for f in segs]
    for im in imgs:
        assert(os.path.isfile(os.path.join(path, im)))

    return imgs, segs


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Get list of image and segmentation files
    image_paths, label_paths = read_files(args.input)
    
    # Get number of spectral channels
    im_hyper, _, _, _ = torchseg.data_loader.read_stiff(image_paths[0], 
                                                        silent=True, 
                                                        rgb_only=False)

    # Compute stats on the dataset
    nchan = im_hyper.shape[2]
    mean = np.zeros((nchan,))
    m2 = np.zeros((nchan,))
    n = 0
    for im_path, label_path in tqdm.tqdm(zip(image_paths, label_paths)):
        # Load image (h, w, c)
        im_hyper, _, _, _ = torchseg.data_loader.read_stiff(image_paths[0], 
                                                            silent=True, 
                                                            rgb_only=False)
        # Update the iterative stats
        h, w, c = im_hyper.shape
        for i in range(h):
            for j in range(w):
                n += 1
                x_n = im_hyper[i, j]
                old_mean = copy.deepcopy(mean)
                mean += (x_n - mean) / n
                m2 += (x_n - old_mean) * (x_n - mean)
    
    # Compute unbiased sample std
    std = np.sqrt(m2 / (n - 1))

    # Save mean and std to file
    meanfile = open(args.mean, 'w')
    stdfile = open(args.std, 'w')
    meanfile.write('[')
    stdfile.write('[')
    for i in range(nchan):
        meanfile.write("{:.6f}".format(mean[i]) + ', ')
        stdfile.write("{:.6f}".format(std[i]) + ', ')
    meanfile.write("]\n")
    stdfile.write("]\n")
    meanfile.close()
    stdfile.close()


if __name__ == '__main__':
    main()
