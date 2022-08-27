"""
@brief   We want to use image-level annotations to learn the segmentation.
         Therefore, we need to make sure that there is at least one image
         (ideally more) where each of the classes is not present. 
        
         The opposite is not a concern, as we know that all the classes are
         present at least in one image.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    27 Apr 2022.
"""

import argparse
import numpy as np
import os
import sys

# My imports
import torchseg.data_loader as dl
import torchseg.utils


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

    # Read parameters
    args = parser.parse_args()
    
    return args


def validate_cmdline_params(args):
    # Input directory must exist
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')
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


def class_in_label(class_name: str, label: np.ndarray) -> bool:
    # Get class index
    idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames
    class2idx = {y: x for x, y in idx2class.items()}
    class_id = class2idx[class_name]

    # Check if there is any pixel of this class
    return bool(np.count_nonzero(label[:, :, class_id]))


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Get list of image and segmentation files
    sys.stdout.write('[INFO] Reading list of input files... ')
    sys.stdout.flush()
    image_paths, label_paths = read_files(args.input)
    print('OK')

    # Initially we assume that all the classes are in all the images
    present_in_all = {v: True for v in dl.OdsiDbDataLoader.OdsiDbDataset.classnames.values()}

    # Loop over the images
    for im_path, label_path in zip(image_paths, label_paths):
        # Real label
        label = dl.OdsiDbDataLoader.LoadImage.read_label(
            label_path).transpose((1, 2, 0))
        
        # Loop over the classes and find out which are not present
        for c in present_in_all:
            # If we know that this class was not present in a previous image
            # we do not need to check this image
            if not present_in_all[c]:
                continue

            # If the class is not present in the image, we write it down
            if not class_in_label(c, label):
                present_in_all[c] = False

    # Make sure that all the classes are missing from at least one image
    success = True 
    for c in present_in_all:
        if present_in_all[c]:
            success = False
            break
    if success:
        print("[OK] You can use image-level presence labels with the ODSI-DB dataset.")
    else:
        print("[ERROR] You cannot use image-level presence labels with the ODSI_DB dataset.")
        print("Check the classes that are present in ALL the images below:")
        print(present_in_all)


if __name__ == '__main__':
    main()
