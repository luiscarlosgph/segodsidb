"""
@brief   Script that reconstructs the RGB images of the ODSI-DB dataset and 
         saves them in output folders, one per camera.
         - Nuance EX (1392x1040 pixels, 450-950nm, 10nm steps).
         - Specim IQ (512x512 pixels, 400-1000nm, 3nm steps).

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
        '-o': 'Path to the output directory (required: True)',
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

    # Create output folders
    nuance_ex_path = os.path.join(args.output, 'nuance_ex')
    specim_iq_path = os.path.join(args.output, 'specim_iq')
    os.mkdir(args.output)
    os.mkdir(nuance_ex_path)
    os.mkdir(specim_iq_path)

    # Get list of image and segmentation files
    image_paths, label_paths = read_files(args.input)
    
    for im_path, label_path in tqdm.tqdm(zip(image_paths, label_paths)):
        im_fname = os.path.splitext(ntpath.basename(im_path))[0]
        label_fname = ntpath.basename(label_path) 

        # Read image
        im_hyper, wl, im_rgb_orig, metadata = dl.read_stiff(im_path, 
            silent=True, rgb_only=False)
        nbands = im_hyper.shape[2]

        # Copy image to the right folder depending on the type of camera, this is to observe 
        # the diferences between the images reconstructed from each of the different cameras
        if nbands == 51:  # Nuance EX
            im_dpath = os.path.join(nuance_ex_path, im_fname + '.jpg')
            im_dpath_orig = os.path.join(nuance_ex_path, im_fname + '_orig.jpg')
            #label_dpath = os.path.join(nuance_ex_path, label_fname) 
        elif nbands == 204:  # Specim IQ
            im_dpath = os.path.join(specim_iq_path, im_fname) + '.jpg'
            im_dpath_orig = os.path.join(specim_iq_path, im_fname + '_orig.jpg')
            #label_dpath = os.path.join(specim_iq_path, label_fname)  
        else:
            raise ValueError('[ERROR] The image {} has {} bands.'.format(im_fname, nbands))

        # Convert hyperspectral image to RGB
        im_rgb = dl.OdsiDbDataLoader.LoadImage.hyper2rgb(im_hyper, wl)
            
        # Save reconstructed RGB image into the output folder
        im_bgr = im_rgb[...,::-1].copy()
        cv2.imwrite(im_dpath, im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # Save RGB provided in ODSI-DB into the output folder
        im_bgr_orig = im_rgb_orig[...,::-1].copy()
        cv2.imwrite(im_dpath_orig, im_bgr_orig, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
