"""
@brief   Script to validate the hyperspectral data of ODSI-DB along with the
         generation of RGB images from hyperspectral data.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    27 Sep 2021.
"""

import argparse
import numpy as np
import os
import tqdm
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.manifold
import multiprocessing
import colour
import sys
import cv2
import scipy

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
        '-o': 'Path to the output directory (required: True)',
        '-n': 'Max. no. of randomly selected pixels for t-SNE (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='PyTorch segmenter.')
    parser.add_argument('-i', '--input', required=True, type=str, 
        help=help('-i'))
    parser.add_argument('-o', '--output', required=True, type=str, 
        help=help('-o'))

    # Read parameters
    args = parser.parse_args()
    
    return args


def validate_cmdline_params(args):
    # Input directory must exist
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')

    # Output directory should not exist, we will create it
    if os.path.exists(args.output):
        raise RuntimeError('[ERROR] Output directory already exists.')

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

    # Create output folder
    os.mkdir(args.output)

    # Get list of image and segmentation files
    sys.stdout.write('[INFO] Reading list of input files... ')
    sys.stdout.flush()
    image_paths, label_paths = read_files(args.input)
    print('OK')

    # Loop over the images
    for im_path, label_path in zip(image_paths, label_paths):
        # Read image
        im_hyper, wl, im_rgb, metadata = torchseg.data_loader.read_stiff(im_path, 
            silent=True, rgb_only=False)
        
        # Convert hyperspectral image to RGB
        fname = os.path.basename(os.path.splitext(im_path)[0])
        sys.stdout.write("[INFO] Converting {} to RGB... ".format(fname))
        image_loader = torchseg.data_loader.OdsiDbDataLoader.LoadImage
        im_recon_rgb = image_loader.hyper2rgb(im_hyper, wl)
        sys.stdout.flush()
        print('OK')

        # Convert RGB image to BGR so that we can save it with OpenCV
        im_recon_bgr = im_recon_rgb[...,::-1] 

        # Save reconstructed RGB image to the output folder  
        im_recon_path = os.path.join(args.output, fname + '_reconstructed.png')
        cv2.imwrite(im_recon_path, im_recon_bgr)

        # Save RGB image provided by the dataset to the output folder  
        im_orig_bgr = im_rgb[...,::-1]
        im_orig_path = os.path.join(args.output, fname + '_original.png')
        cv2.imwrite(im_orig_path, im_orig_bgr)


if __name__ == '__main__':
    main()
