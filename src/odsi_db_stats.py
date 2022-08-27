"""
@brief   Script that show some ODSI-DB stats that are relevant for the paper:
         - Number of images of 51 bands
         - Number of images of 204 bands
         - Number of pixels per class
         - Number of images in which a class is annotated

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    3 Sep 2021.
"""

import argparse
import numpy as np
import os
import shutil
import sys
import tqdm
import json
import random

# My imports
import torchseg.data_loader as dl
import torchseg.utils


# Get dictionary from class index to class string, e.g. 0 -> u'Attached gingiva'
classnames = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames


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
    """@brief Ensure that input directory exits."""
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')

    return args


def read_files(path):
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


def num_images_per_camera(images):
    """@brief Compute number of images per camera."""
    bands_51 = 0
    bands_204 = 0
    for im_path in images:
        im, wl, im_rgb, metadata = dl.read_stiff(im_path, silent=True,
                                                 rgb_only=False)
        if im.shape[2] == 51:
            bands_51 += 1
        elif im.shape[2] == 204:
            bands_204 += 1
        else:
            raise ValueError('[ERROR] The image ' + im_path + ' has ' \
                + str(im.shape[2]) + ' bands.')
    return bands_51, bands_204


def get_latex_table(label, columns, data_pix, data_im, caption):
    """
    @brief Get a table with the stats for each class in the dataset.

    @param[in]  label      Latex label of the table.
    @param[in]  columns    List of column names.
    @param[in]  data_pix   Class -> number of pixels.
    @param[in]  data_im    Class -> number of images.
    @param[in]  caption    Caption of the figure. 

    @returns the string with the latex code.
    """
    latex = "\\begin{table*}[htb!]\n"
    latex += "    \centering\n"
    latex += "    \caption{" + caption + "}\n"
    latex += "    \\vspace{0.2cm}\n"
    latex += "    \\begin{tabular}{lrr}\n"
    latex += "        \\hline\n"
    latex += "        \multicolumn{1}{c}{\\bfseries " + columns[0] + "} &\n"
    latex += "        \multicolumn{1}{c}{\\bfseries " + columns[1] + "} &\n"
    latex += "        \multicolumn{1}{c}{\\bfseries " + columns[2] + "} \\\\ \n"
    latex += "        \hline\n"


    # Print rows here
    data_pix = dict(sorted(data_pix.items(), key=lambda item: item[1], 
                           reverse=True))
    for k in data_pix:
        latex += '        ' + k + ' & ' + str(data_pix[k]) + ' & ' + str(data_im[k]) + " \\\\ \n"

    latex += "    \end{tabular}\n"
    latex += "    \\vspace{0.2cm}\n"
    latex += "    \label{tab:" + label + "}\n"
    latex += "\end{table*}"
    
    return latex


def num_pixels_per_class(images, labels):
    """@brief Compute the number of pixels belonging to each class."""
    class2numpix = {v: 0 for k, v \
        in dl.OdsiDbDataLoader.OdsiDbDataset.classnames.items()}
    class2numim = {v: 0 for k, v \
        in dl.OdsiDbDataLoader.OdsiDbDataset.classnames.items()}
    for label_path in labels: 
        label = torchseg.data_loader.read_mtiff(label_path)
        for c in label:
            class2numim[c] += 1
            class2numpix[c] += np.count_nonzero(label[c])
    return class2numpix, class2numim


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Get list of image and segmentation files
    images, labels = read_files(args.input)
    
    # Compute the number of pixels per class
    class2numpix, class2numim = num_pixels_per_class(images, labels)
    print(get_latex_table('tab:class_to_numpix', 
                          ['Class', 'Number of pixels', 'Number of images'],
                          class2numpix, class2numim, 
                          'Number of pixels per class.'))
   
    # Compute the number of images recorded by Nuance EX (51 bannds) and
    # the number of images recorded by Specim IQ (204 bands)
    bands_51, bands_204 = num_images_per_camera(images)
    print('Number of images with 51 bands:', bands_51)
    print('Number of images with 204 bands:', bands_204)
    print('Total number of annotated images:', len(images))


if __name__ == '__main__':
    main()
