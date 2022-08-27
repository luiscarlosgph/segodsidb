"""
@brief   Script that generates a t-SNE of all the pixels in the ODSI-DB dataset,
         comparing RGB pixels with hyperspectral pixels.

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
import matplotlib
import matplotlib.pyplot as plt
import sklearn.manifold
import multiprocessing
import scipy.spatial
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
        '-o': 'Path to the output directory (required: True)',
        '-n': 'Max. no. of selected pixels for t-SNE (required: True)',
        '-r': 'Use reconstructed RGB images (required: True)',
        '-e': 'Convert Specim IQ images to Nuance EX style of 51 bands (required: True)',
        '-v': 'Use only the visible spectrum (required: True)',
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
    parser.add_argument('-n', '--npixels', required=True, type=int, 
        help=help('-n'))
    parser.add_argument('-r', '--recon', required=True, type=bool,
        help=help('-r'))
    parser.add_argument('-v', '--visible', required=True, type=bool,
        help=help('-v'))
    parser.add_argument('-e', '--nuance', required=True, type=str,
        help=help('-e'))

    # Read parameters
    args = parser.parse_args()
    
    return args


def validate_cmdline_params(args):

    # Input directory must exist
    if not os.path.isdir(args.input):
        raise ValueError('[ERROR] Input directory does not exist.')

    # Output directory should not exist, we will create it
    if os.path.exists(args.output):
        raise ValueError('[ERROR] Output directory already exists.')

    if args.nuance not in ['none', 'nearest', 'linear']:
        raise ValueError('[ERROR] Specim -> Nuance conversion mode not recognised.')

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


def specim2nuance(im_specim, specim_wl=None, nuance_wl=None, mode='linear'):
    """
    @brief Convert a hyperspectral image from the Specim IQ camera (204 bands)
           to the Nuance EX (51 bands). 
    @param[in]  im_specim  Numpy ndarray, shape (h, w, c).
    @param[in]  specim_wl  Array of wavelengths of the Specim IQ bands.
    @param[in]  nuance_wl  Array of wavelengths of the Nuance EX bands. 
    @param[in]  mode       Two modes are available: 'nearest' and 'linear'.
                               nearest: the band with nearest wavelength of 
                                        the Specim IQ is used.
                               linear:  linear interpolation of the two 
                                        surrounding wavelengths.
    """ 
    # Get an array of the wavelengths of each camera
    if specim_wl is None: 
        specim_wl = np.array([ 
	    397.32000732,  400.20001221,  403.08999634,  405.97000122,  408.8500061 ,
	    411.73999023,  414.63000488,  417.51998901,  420.3999939 ,  423.29000854,
	    426.19000244,  429.07998657,  431.97000122,  434.86999512,  437.76000977,
	    440.66000366,  443.55999756,  446.45001221,  449.3500061 ,  452.25      ,
	    455.16000366,  458.05999756,  460.95999146,  463.86999512,  466.76998901,
	    469.67999268,  472.58999634,  475.5       ,  478.41000366,  481.32000732,
	    484.23001099,  487.14001465,  490.05999756,  492.97000122,  495.89001465,
	    498.79998779,  501.72000122,  504.64001465,  507.55999756,  510.48001099,
	    513.40002441,  516.33001709,  519.25      ,  522.17999268,  525.09997559,
	    528.0300293 ,  530.96002197,  533.89001465,  536.82000732,  539.75      ,
	    542.67999268,  545.61999512,  548.54998779,  551.48999023,  554.42999268,
	    557.35998535,  560.29998779,  563.23999023,  566.17999268,  569.11999512,
	    572.07000732,  575.01000977,  577.96002197,  580.90002441,  583.84997559,
	    586.79998779,  589.75      ,  592.70001221,  595.65002441,  598.59997559,
	    601.54998779,  604.51000977,  607.46002197,  610.41998291,  613.38000488,
	    616.34002686,  619.29998779,  622.26000977,  625.2199707 ,  628.17999268,
	    631.15002441,  634.10998535,  637.08001709,  640.03997803,  643.01000977,
	    645.97998047,  648.95001221,  651.91998291,  654.89001465,  657.86999512,
	    660.84002686,  663.80999756,  666.78997803,  669.77001953,  672.75      ,
	    675.72998047,  678.71002197,  681.69000244,  684.66998291,  687.65002441,
	    690.64001465,  693.61999512,  696.60998535,  699.59997559,  702.58001709,
	    705.57000732,  708.57000732,  711.55999756,  714.54998779,  717.53997803,
	    720.53997803,  723.5300293 ,  726.5300293 ,  729.5300293 ,  732.5300293 ,
	    735.5300293 ,  738.5300293 ,  741.5300293 ,  744.5300293 ,  747.53997803,
	    750.53997803,  753.54998779,  756.55999756,  759.55999756,  762.57000732,
	    765.58001709,  768.59997559,  771.60998535,  774.61999512,  777.64001465,
	    780.65002441,  783.66998291,  786.67999268,  789.70001221,  792.7199707 ,
	    795.73999023,  798.77001953,  801.78997803,  804.80999756,  807.84002686,
	    810.85998535,  813.89001465,  816.91998291,  819.95001221,  822.97998047,
	    826.01000977,  829.03997803,  832.07000732,  835.10998535,  838.14001465,
	    841.17999268,  844.2199707 ,  847.25      ,  850.28997803,  853.33001709,
	    856.36999512,  859.41998291,  862.46002197,  865.5       ,  868.54998779,
	    871.59997559,  874.64001465,  877.69000244,  880.73999023,  883.78997803,
	    886.84002686,  889.90002441,  892.95001221,  896.01000977,  899.05999756,
	    902.11999512,  905.17999268,  908.23999023,  911.29998779,  914.35998535,
	    917.41998291,  920.47998047,  923.54998779,  926.60998535,  929.67999268,
	    932.73999023,  935.80999756,  938.88000488,  941.95001221,  945.02001953,
	    948.09997559,  951.16998291,  954.23999023,  957.32000732,  960.40002441,
	    963.4699707 ,  966.54998779,  969.63000488,  972.71002197,  975.78997803,
	    978.88000488,  981.96002197,  985.04998779,  988.13000488,  991.2199707 ,
	    994.30999756,  997.40002441, 1000.48999023, 1003.58001709,
        ])
    if nuance_wl is None:
        nuance_wl = np.linspace(450, 950, 51)
    
    # If we get a Nuance EX image, we don't need to convert anything
    if im_specim.shape[2] == 51:
        return im_specim, nuance_wl 
    assert(im_specim.shape[2] == 204)
    
    # Compute the distance between wavelengths of the two cameras
    dist = scipy.spatial.distance.cdist(specim_wl.reshape((204, 1)), 
        nuance_wl.reshape((51, 1)), metric='euclidean')
    
    # Compute the nearest nuance wavelength for each specim band
    nearest = np.argmin(dist, axis=0) 
    
    # Synthesize Nuance-like image using the nearest bands
    h, w, _ = im_specim.shape
    im_nuance = np.empty((h, w, nuance_wl.shape[0]), dtype=np.float32)
    if mode == 'nearest':
        for i in range(nuance_wl.shape[0]): 
            im_nuance[:, :, i] = im_specim[:, :, nearest[i]].copy()
    elif mode == 'linear':
        for i in range(nuance_wl.shape[0]):
            # Find the bands before and after
            if specim_wl[nearest[i]] > nuance_wl[i]:
                j = nearest[i] - 1 
                k = nearest[i] 
            else:
                j = nearest[i] 
                k = nearest[i] + 1 

            # Compute the weights of the bands before and after
            d1 = nuance_wl[i] - specim_wl[j]
            d2 = specim_wl[k] - nuance_wl[i]
            norm = d1 + d2
            w1 = d2 / norm
            w2 = d1 / norm
            
            # Add interpolated channel to the synthetic Nuance EX image
            im_nuance[:, :, i] = w1 * im_specim[:, :, j] + w2 * im_specim[:, :, k]
    else:
        raise ValueError('[ERROR] Unknown interpolation mode.')
    
    assert(im_nuance.shape == (h, w, 51))
    return im_nuance, nuance_wl


def process_image(im_path: str, label_path: str, no_pixels: int, 
        rgb_recon: bool, visible: bool, nuance_recon: str):
    """
    @brief Get the annotated pixels from an image of the ODSI-DB dataset.
    @param[in]  im_path      Path to the image file.
    @param[in]  label_path   Path to the annotation file. 
    @param[in]  no_pixels    Number of annotated pixels to be sampled from the
                             image.
    @param[in]  rgb_recon    Reconstruct RGB image from hyperspectral.
    @param[in]  visible      Only use visible wavelengths.
    @param[in]  nuance_recon Convert Specim IQ images to the Nuance EX style
                             of 51 bands.
    """
    # Read raw hyperspectral image
    im_hyper, wl, im_rgb, metadata = torchseg.data_loader.read_stiff(im_path, 
            silent=True, rgb_only=False)
    
    # Convert image to Nuance EX (in case it is Specim IQ)
    if nuance_recon is not None:
        im_hyper, wl = specim2nuance(im_hyper, mode=nuance_recon)
    
    # Remove the non-visible part of the spectrum if requested
    if visible:
        im_hyper, wl = torchseg.data_loader.OdsiDbDataLoader.LoadImage.filter_bands(im_hyper, wl,
                                                                                    380., 740.)
    # Reconstruct RGB image if requested
    if rgb_recon:
        im_rgb = torchseg.data_loader.OdsiDbDataLoader.LoadImage.hyper2rgb(im_hyper, wl)

    # Read label
    label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
        label_path).transpose((1, 2, 0))
    
    # Make sure that image and label have the same height and width
    assert(label.shape[2] == 35)  # Labels are supposed to have 35 classes 
    assert(im_hyper.shape[0] == label.shape[0])
    assert(im_hyper.shape[1] == label.shape[1])
    assert(im_rgb.shape[0] == label.shape[0])
    assert(im_rgb.shape[1] == label.shape[1])

    # Collect the labelled pixels
    ann = np.where(np.sum(label, axis=2) == 1)
    ann = list(zip(ann[0], ann[1]))

    # Downsample the number of pixels
    random.shuffle(ann)
    ann = ann[:no_pixels] 

    # Save the randomly selected pixels
    hyper_pixels = []
    rgb_pixels = []
    labels = []
    for i, j in ann:
        hyper_pixels.append(im_hyper[i, j, :].tolist())
        rgb_pixels.append(im_rgb[i, j, :].tolist())
        labels.append(np.argmax(label[i, j, :]))
    return hyper_pixels, rgb_pixels, labels


def get_pixels(image_paths, label_paths, no_pixels=1000000, rgb_recon=False,
        visible=False, nuance_recon=False):
    """
    @brief Get all the labelled pixels of the ODSI-DB dataset.
    @param[in]  image_paths   List of image paths.
    @param[in]  label_paths   List of label paths in sync with the list of 
                              images.
    @param[in]  rgb_recon     True if you want to reconstruct the RGB images 
                              from the hyperspectral images and not use the
                              ones already provided in ODSI-DB.
    @param[in]  visible       True if you want to use just the visible part
                              of the spectrum.
    @param[in]  nuance_recon  Convert Specim IQ image to Nuance EX style. 
    """
    assert(len(image_paths) == len(label_paths))
    nimages = len(image_paths)
    ppi = no_pixels // nimages 

    retval = None
    with multiprocessing.Pool() as pool:
        data_input = list(zip(image_paths, label_paths, [ppi] * nimages, 
                              [rgb_recon] * nimages, [visible] * nimages,
                              [nuance_recon] * nimages))
        retval = pool.starmap(process_image, data_input)[0]
    #hyper_pixels, rgb_pixels = retval[0]
    #labels = retval[1]

    return retval


def plot(rgb_embed, hyper_embed, labels):
    """
    @brief Plots the t-SNE of RGB and hyperspectral pixels.
    @param[in]  rgb_embed    Embedding of RGB pixels (N, 2).
    @param[in]  hyper_embed  Embedding of hyperspectral pixels (N, 2).
    @param[in]  labels       List of strings (N,). 
    @returns a tuple (fig, axes, lgd) where fig and ax are the usual matplotlib
             objects and lgd is the fig.legend.
    """
    # Prepare data to plot with seaborn
    rgb_data = {'y': rgb_embed[:, 0], 'x': rgb_embed[:, 1], 'label': labels}
    hyper_data = {'y': hyper_embed[:, 0], 'x': hyper_embed[:, 1], 'label': labels}

    # Create figure
    sns.set()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Generate subplots
    palette = dict(zip(idx2class.values(), 
                       sns.color_palette('husl', n_colors=len(idx2class.values()))))
    #                   sns.color_palette(n_colors=len(idx2class.values()))))
    rgb_plot = sns.scatterplot(ax=axes[0], data=rgb_data, x='x', y='y', 
                               hue='label', s=5)
    hyper_plot = sns.scatterplot(ax=axes[1], data=hyper_data, x='x', y='y', 
                                 hue='label', s=5)

    # Make plots beautiful
    ymax = max(rgb_embed[:, 0].max(), hyper_embed[:, 0].max())
    ymin = min(rgb_embed[:, 0].min(), hyper_embed[:, 0].min())
    xmax = max(rgb_embed[:, 1].max(), hyper_embed[:, 1].max())
    xmin = min(rgb_embed[:, 1].min(), hyper_embed[:, 1].min())
    handles, figlabels = fig.axes[-1].get_legend_handles_labels()
    lgd = fig.legend(handles, figlabels, loc='lower center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=10)
    #lgd = fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[0].set_title('RGB', pad=20)
    #rgb_plot.set(title='RGB')
    rgb_plot.set(xlabel=None)
    rgb_plot.set(ylabel=None)
    rgb_plot.set(ylim=[ymin, ymax])
    rgb_plot.set(xlim=[xmin, xmax])
    rgb_plot.set(xticklabels=[])
    rgb_plot.set(yticklabels=[])
    rgb_plot.set(facecolor='white')
    rgb_plot.get_legend().remove()
    axes[1].set_title('Hyperspectral', pad=20)
    #hyper_plot.set(title='Hyperspectral')
    hyper_plot.set(xlabel=None)
    hyper_plot.set(ylabel=None)
    hyper_plot.set(ylim=[ymin, ymax])
    hyper_plot.set(xlim=[xmin, xmax])
    hyper_plot.set(xticklabels=[])
    hyper_plot.set(yticklabels=[])
    hyper_plot.set(facecolor='white')
    hyper_plot.get_legend().remove()

    return fig, axes, lgd 


def savefig(fig: matplotlib.figure.Figure, lgd: matplotlib.legend.Legend, 
        path: str):
    """
    @brief Save figure to file.
    @param[in]  fig   Seaborn figure.
    @param[in]  lgd   Seaborn legend. This is needed because we want to place
                      the legend outside of the plotting area. To do this, we
                      need to pass the legend to the 'bbox_extra_artists' 
                      argument of fig.savefig().
    @param[in]  path  Path to the output file where the figure will be saved.
    """
    plt.tight_layout()
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def specimen_iq_visble_rgb_metric(pixa, pixb):
    wl = np.array([ 
        397.32000732, 400.20001221, 403.08999634, 405.97000122, 408.8500061,
        411.73999023, 414.63000488, 417.51998901, 420.3999939 , 423.29000854,
        426.19000244, 429.07998657, 431.97000122, 434.86999512, 437.76000977,
        440.66000366, 443.55999756, 446.45001221, 449.3500061,  452.25,
        455.16000366, 458.05999756, 460.95999146, 463.86999512, 466.76998901,
        469.67999268, 472.58999634, 475.5,        478.41000366, 481.32000732,
        484.23001099, 487.14001465, 490.05999756, 492.97000122, 495.89001465,
        498.79998779, 501.72000122, 504.64001465, 507.55999756, 510.48001099,
        513.40002441, 516.33001709, 519.25,       522.17999268, 525.09997559,
        528.0300293,  530.96002197, 533.89001465, 536.82000732, 539.75,
        542.67999268, 545.61999512, 548.54998779, 551.48999023, 554.42999268,
        557.35998535, 560.29998779, 563.23999023, 566.17999268, 569.11999512,
        572.07000732, 575.01000977, 577.96002197, 580.90002441, 583.84997559,
        586.79998779, 589.75,       592.70001221, 595.65002441, 598.59997559,
        601.54998779, 604.51000977, 607.46002197, 610.41998291, 613.38000488,
        616.34002686, 619.29998779, 622.26000977, 625.2199707 , 628.17999268,
        631.15002441, 634.10998535, 637.08001709, 640.03997803, 643.01000977,
        645.97998047, 648.95001221, 651.91998291, 654.89001465, 657.86999512,
        660.84002686, 663.80999756, 666.78997803, 669.77001953, 672.75,
        675.72998047, 678.71002197, 681.69000244, 684.66998291, 687.65002441,
        690.64001465, 693.61999512, 696.60998535, 699.59997559, 702.58001709,
        705.57000732, 708.57000732, 711.55999756, 714.54998779, 717.53997803,
        720.53997803, 723.5300293,  726.5300293,  729.5300293,  732.5300293,
        735.5300293,  738.5300293,
    ])

    # Create image with the two pixels
    im_hyper = np.dstack((pixa, pixb)).transpose((0, 2, 1))

    # Convert image to RGB
    image_loader = torchseg.data_loader.OdsiDbDataLoader.LoadImage
    im = image_loader.hyper2rgb(im_hyper, wl)

    # Compute Euclidean distance in RGB
    dist = np.linalg.norm(im[0, 0] - im[0, 1])

    return dist


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Create output folder
    os.mkdir(args.output)

    # Get list of image and segmentation files
    image_paths, label_paths = read_files(args.input)
    
    # FIXME: this is for debugging
    #image_paths = image_paths[:1]
    #label_paths = label_paths[:1]

    # Read pixels from the dataset
    args.nuance = None if args.nuance == 'none' else args.nuance
    hyper_pixels, rgb_pixels, labels = get_pixels(image_paths, label_paths, 
                                                  args.npixels,
                                                  rgb_recon=args.recon,
                                                  visible=args.visible,
                                                  nuance_recon=args.nuance)

    #rgb_pixels = np.array(rgb_pixels)
    #hyper_pixels = np.array(hyper_pixels)
    #assert(rgb_pixels.shape[0] == hyper_pixels.shape[0])
    assert(len(hyper_pixels) == len(rgb_pixels))
    assert(len(rgb_pixels) == len(labels))

    # Convert labels from class indices to class names
    labels = [idx2class[l] for l in labels]

    # t-SNE
    for perp in list(range(10, 210, 10)):
        print("Running t-SNE with perplexity = {}...".format(perp))
        rgb_tsne = sklearn.manifold.TSNE(n_components=2, verbose=0, 
                                         perplexity=perp, 
                                         n_iter=1000, learning_rate=200, 
                                         init='pca')
        hyper_tsne = sklearn.manifold.TSNE(n_components=2, verbose=0, 
                                           perplexity=perp, 
                                           n_iter=1000, learning_rate=200, 
                                           init='pca')
        rgb_embed = rgb_tsne.fit_transform(rgb_pixels)
        hyper_embed = hyper_tsne.fit_transform(hyper_pixels)
        
        # Save embeddings
        with open(os.path.join(args.output, "rgb_embed_perplexity_{}.npy".format(perp)), 'wb') as f:
            np.save(f, rgb_embed)

        with open(os.path.join(args.output, "hyper_embed_perplexity_{}.npy".format(perp)), 'wb') as f:
            np.save(f, hyper_embed)

        # Plot RGB and hyperspectral t-SNE subplots
        fig, axes, lgd = plot(rgb_embed, hyper_embed, labels)

        # Save figure inside the output folder
        savefig(fig, lgd, os.path.join(args.output, "figure_perplexity_{}.png".format(perp)))

if __name__ == '__main__':
    main()
