"""
@brief   Compute the channel std of the 204 interpolated bands from
         400-1000nm.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    8 Jul 2022.
"""

import argparse
import numpy as np
import os
import ntpath
import shutil
import tqdm
import cv2
import sys

# My imports
import torchseg.data_loader as dl
import torchseg.utils


# Get dictionary from class index to class string, 
# e.g. 0 -> u'Attached gingiva'
idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames


def help(short_option):
    """
    @returns The string with the help information for each command 
             line option.
    """
    help_msg = {
        '-i': 'Path to the root of the ODSI-DB dataset (required: True)',
        '-m': 'Channel means normalised to the range [0, 1] (required: True)',
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
    image_paths, _ = read_files(args.input)
    
    # Read means 
    mean_sq = np.array(args.means) * np.array(args.means)
    
    # Hyperspectral std computation stuff
    min_wl = 400
    max_wl = 1000
    nbands = 204
    new_wl = np.linspace(min_wl, max_wl, nbands)
    hyper_sums = np.zeros((new_wl.shape[0],), dtype=np.float64)
    pixel_count = 0
    image_count = 0
    interp_func = dl.OdsiDbDataLoader.LoadImage.interp_spectra

    # Get the indices of the wavelengths that are shared among both cameras
    inside_indices = np.array([idx for idx, x in enumerate(new_wl.tolist()) \
        if x > 450. and x < 950.])
    inside_wl = new_wl[inside_indices]

    # Print interval
    interval = 10

    # Loop over the images
    for im_path in tqdm.tqdm(image_paths):
        im_fname = os.path.splitext(ntpath.basename(im_path))[0]

        # Read image
        im_hyper, wl, _, _ = dl.read_stiff(im_path, silent=True, 
                                           rgb_only=False)
        npixels = im_hyper.shape[0] * im_hyper.shape[1]

        # Handle Nuance EX images
        if im_hyper.shape[2] == 51:
            # Interpolate to bands in the range [450-950]
            im_hyper_inside = interp_func(im_hyper, wl, inside_wl)

            # Extrapolate to 204 bands
            im_hyper = np.empty((im_hyper_inside.shape[0], 
                                 im_hyper_inside.shape[1],
                                 nbands), 
                                 dtype=im_hyper_inside.dtype)
            j = 0
            for i in range(nbands):
                if i < np.min(inside_indices):
                    im_hyper[:, :, i] = im_hyper_inside[:, :, 0]
                elif i > np.max(inside_indices):
                    im_hyper[:, :, i] = im_hyper_inside[:, :, -1]
                else:
                    im_hyper[:, :, i] = im_hyper_inside[:, :, j] 
                    j += 1

        # Handle Specim IQ images
        else:
            im_hyper = interp_func(im_hyper, wl, new_wl)

        # Update std computation
        im_hyper_sq = im_hyper * im_hyper
        for i in range(hyper_sums.shape[0]):
            hyper_sums[i] += (im_hyper_sq[:, :, i] - mean_sq[i]).sum()

        # Update counters
        pixel_count += npixels
        image_count += 1
        
        # Print intermediate stats
        if image_count % interval == 0:
            print('Image count:', image_count)
            sys.stdout.write('All hyper 400-1000nm (204 bands) std: [')
            for i in range(nbands):
                sys.stdout.write(str(hyper_sums[i] / pixel_count))
                if i < nbands - 1:
                    sys.stdout.write(', ')
            sys.stdout.write(']\n')
                
    # Print final stats
    print('Image count:', image_count)
    sys.stdout.write('All hyper 400-1000nm (204 bands) std: [')
    for i in range(nbands):
        sys.stdout.write(str(hyper_sums[i] / pixel_count))
        if i < nbands - 1:
            sys.stdout.write(', ')
    sys.stdout.write(']')


if __name__ == '__main__':
    main()
