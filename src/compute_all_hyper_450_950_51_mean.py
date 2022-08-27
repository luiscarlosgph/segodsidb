"""
@brief   Compute the channel mean of the 51 interpolated bands from 
         450-950nm.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    21 Jun 2022.
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


# Get dictionary from class index to class string, e.g. 0 -> u'Attached gingiva'
idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the ODSI-DB dataset (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='PyTorch segmenter.')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))

    # Read parameters
    args = parser.parse_args()
    
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
    
    # Hyperspectral mean computation stuff
    nbands = 51 
    new_wl = np.linspace(450, 950, nbands)
    hyper_sums = np.zeros((nbands,), dtype=np.float64)
    pixel_count = 0
    image_count = 0
    
    # Print interval 
    interval = 10

    # Loop over the images
    for im_path in tqdm.tqdm(image_paths):
        im_fname = os.path.splitext(ntpath.basename(im_path))[0]

        # Read image
        im_hyper, wl, _, _ = dl.read_stiff(im_path, silent=True, 
                                           rgb_only=False)

        # Interpolate hyperspectral image to the 450-950nm range (51 bands)
        interp_func = dl.OdsiDbDataLoader.LoadImage.interp_spectra
        im_hyper = interp_func(im_hyper, wl, new_wl)

        # Update average computation
        for i in range(nbands):
            hyper_sums[i] += im_hyper[:, :, i].sum()
        pixel_count += im_hyper.shape[0] * im_hyper.shape[1]
        image_count += 1
        
        # Print intermediate stats
        if image_count % interval == 0:
            print('Image count:', image_count)
            sys.stdout.write('All hyper 450-950nm (51 bands) average: [')
            for i in range(nbands):
                sys.stdout.write(str(hyper_sums[i] / pixel_count))
                if i < nbands - 1:
                    sys.stdout.write(', ')
            sys.stdout.write(']\n')
                
    # Print final stats
    print('Image count:', image_count)
    sys.stdout.write('All hyper 450-950nm (51 bands) average: [')
    for i in range(nbands):
        sys.stdout.write(str(hyper_sums[i] / pixel_count))
        if i < nbands - 1:
            sys.stdout.write(', ')
    sys.stdout.write(']')


if __name__ == '__main__':
    main()
