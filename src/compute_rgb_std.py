"""
@brief   Compute the std of the RGB images in the ODSI-DB dataset,
         normalised to the range [0, 1].

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    16 Feb 2022.
"""

import argparse
import numpy as np
import os
import ntpath
import shutil
import tqdm
import cv2

# My imports
import torchseg.data_loader as dl
import torchseg.utils


# Get dictionary from class index to class string, e.g. 0 -> u'Attached gingiva'
idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the ODSI-DB dataset (required: True)',
        '-m': 'RGB means normalised to the range [0, 1] (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='PyTorch segmenter.')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))
    parser.add_argument('-m', '--means', required=True, type=str, 
                        help=help('-m'))

    # Read parameters
    args = parser.parse_args()
    args.means = eval(args.means)
    
    return args


def validate_cmdline_params(args):
    """
    @brief Make sure that input directory exists.
    @returns nothing.
    """
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')


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

    # Read means
    red_mean_sq = args.means[0] * args.means[0]
    green_mean_sq = args.means[1] * args.means[1]
    blue_mean_sq = args.means[2] * args.means[2]

    # RGB average 
    red_sum     = 0.0
    green_sum   = 0.0
    blue_sum    = 0.0
    pixel_count = 0
    image_count = 0
    
    # Print interval 
    interval = 10

    # Loop over the images
    for im_path in tqdm.tqdm(image_paths):
        im_fname = os.path.splitext(ntpath.basename(im_path))[0]

        # Read image
        im_hyper, wl, im_rgb_orig, metadata = dl.read_stiff(im_path, 
                                                            silent=True, 
                                                            rgb_only=False)
        nbands = im_hyper.shape[2]

        # Convert hyperspectral image to RGB (H, W, 3)
        im_rgb = dl.OdsiDbDataLoader.LoadImage.hyper2rgb(im_hyper, wl)

        # Normalise the image in the range [0., 1.]
        im_rgb = im_rgb.astype(np.float64) / 255.

        # Update std computation
        im_rgb_sq = im_rgb * im_rgb
        red_sum     += (im_rgb_sq[:, :, 0] - red_mean_sq).sum()
        green_sum   += (im_rgb_sq[:, :, 1] - green_mean_sq).sum()
        blue_sum    += (im_rgb_sq[:, :, 2] - blue_mean_sq).sum()
        pixel_count += im_rgb.shape[0] * im_rgb.shape[1]
        image_count += 1
        
        # Print intermediate stats
        if image_count % interval == 0:
            print('Image count:', image_count)
            print('Red std:', np.sqrt(red_sum / pixel_count))
            print('Green std:', np.sqrt(green_sum / pixel_count))
            print('Blue std:', np.sqrt(blue_sum / pixel_count))
    
    # Print final stats
    print('Image count:', image_count)
    print('Red std:', np.sqrt(red_sum / pixel_count))
    print('Green std:', np.sqrt(green_sum / pixel_count))
    print('Blue std:', np.sqrt(blue_sum / pixel_count))


if __name__ == '__main__':
    main()
