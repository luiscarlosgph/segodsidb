"""
@brief   Script that generates a convenient split of training and testing
         images for the ODSI-DB dataset. Then you can split the training fold
         into training and validation however you want, but this is expected to 
         be done in your dataloader.

@details A convenient dataset must have at least pixels of all the classes in all 
         the splits. An additional desirable feature is to have an even balance 
         of pixels per class in all the splits.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    1 Jun 2021.
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
import torchseg.data_loader
import torchseg.utils


# Get dictionary from class index to class string, e.g. 0 -> u'Attached gingiva'
classnames = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the ODSI-DB dataset (required: True)',
        '-o': 'Path to the output folder where the splits will be saved \
               (required: True)',
        '-t': 'Portion of images to be used for training (default: 0.9)',
        '-f': 'Number of cross-validation folds (default: 1).',
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
    parser.add_argument('-t', '--train', default=0.9, type=float, 
        help=help('-t'))
    parser.add_argument('-f', '--folds', default=1, type=int, 
        help=help('-f'))

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


def read_image(self, path, bands=np.linspace(450, 950, 51)):
    """
    @param[in]  path  String with the path to the hyperspectral image.
    @returns an image with shape (C, H, W) containing values in the range
             [0, 1].
    """
    # Read raw TIFF file
    raw_tiff = torchseg.data_loader.read_stiff(path, silent=True,
        rgb_only=False)
    
    # In the ODSI-DB dataset there are images with 51 bands
    # (450-950nm, both inclusive) and images with 204 bands
    # (400-1000nm)
    
    # Create empty image
    h = raw_tiff[0].shape[0]
    w = raw_tiff[0].shape[1]
    chan = bands.shape[0]
    im = np.empty((chan, h, w), dtype=np.float32)
    
    # Populate image with the wanted bands (bands parameter of this
    # function)
    raw_im = raw_tiff[0].transpose((2, 0, 1)).astype(np.float32)
    wavelengths = raw_tiff[1]
    for idx, wl in enumerate(bands.tolist()):
        matched_idx = np.abs(wavelengths - wl).argmin()
        im[idx, ...] = raw_im[matched_idx, ...].copy()
    
    return im


def read_label(path):
    """
    @param[in]  path  String with the path to the annotation file.
    @returns a numpy.ndarray with shape (C, H, W) containing values in the range
             [0, 1]. 
    """ 
    raw_tiff = torchseg.data_loader.read_mtiff(path)

    # Get the shape of the labels (which should be identical to the
    # height and width of the image)
    shape = raw_tiff[list(raw_tiff.keys())[0]].shape
    
    # Create tensor of labels
    n_classes = len(classnames)
    label = np.zeros((n_classes, *shape), dtype=np.float32)
    
    # Build dictionaries to convert quickly from index to class name
    # and vice versa
    idx2class = classnames
    class2idx = {y: x for x, y in idx2class.items()}
    
    # Populate the binary array for each class
    for k, gt in raw_tiff.items():
        if k not in class2idx:
            raise ValueError('[ERROR] Unknown <' + k + '> label.')
        
        # Find the index of the class with name 'k'
        idx = class2idx[k]
    
        # Set the label for the class with name 'k' 
        label[idx] = gt.astype(np.float32)
    
    return label


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


def generate_random_split(images, labels, train_ratio, 
        force_train=['Fibroma', 'Makeup', 'Malignant lesion']):
    """
    @brief Generates a split of the data for training and testing.
    @details Every element in the list 'images' must correspond to a label in the
             same index of the list 'labels'.

    @param[in]  images       List of paths to the images.
    @param[in]  labels       List of paths to the labels.
    @param[in]  train_ratio  Portion of the images that will go for training. 
    @param[in]  force_train  List of classes. Images containing pixels of these
                             classes will be automatically put in the training
                             set. The reason for this is that there is only one
                             image in the dataset containing pixels of these
                             classes.

    @returns a dictionary of dictionaries. Keys 'train' and 'test' in the
             upper level. Keys 'images' and 'labels' in the lower level.
    """
    np.random.seed()
    class2idx = {y: x for x, y in classnames.items()}
    split_dic = {
        'train': {
            'images': [],
            'labels': [],
        },
        'test': {
            'images': [],
            'labels': [],
        }
    }
    

    # Make a list of the indices of all the images            
    remaining = list(range(len(images))) 

    # Make a dictionary from each class to the indices of the images that
    # contain pixels of such class
    class2img = {c: [] for c in range(len(classnames))}
    for im_idx in remaining:
        label_path = labels[im_idx] 
        label = read_label(label_path).sum(axis=(1, 2)).astype(np.int64)
        for c in label.nonzero()[0].tolist():
            class2img[c].append(im_idx)

    # Put an image of each class in the training set
    for c in class2img:
        random.shuffle(class2img[c])
        im_idx = class2img[c][0]
        if im_idx in remaining: 
            split_dic['train']['images'].append(images[im_idx])
            split_dic['train']['labels'].append(labels[im_idx])
            remaining.remove(im_idx)

    # Split the rest of the images according to the given probability
    while remaining:
        im_idx = remaining.pop()
        dst = 'train' if np.random.binomial(1, train_ratio) else 'test'
        split_dic[dst]['images'].append(images[im_idx])
        split_dic[dst]['labels'].append(labels[im_idx])

    return split_dic


def validate_random_split(split_dic, 
        force_train=['Fibroma', 'Makeup', 'Malignant lesion']):
    """
    @brief Checks that all the classes are represented in both training and
           testing splits.
    @param[in]  split_dic  Dictionary of dictionaries. Keys 'train' and 'test' 
                           in the upper level. Keys 'images' and 'labels' in the 
                           lower level.
    @returns True if the split is good. Otherwise, returns False.
    """
    classes = len(classnames)
    train_classes = np.zeros((classes,), dtype=np.int64)
    test_classes = np.zeros((classes,), dtype=np.int64)
    
    # Find the classes present in the training set
    for label_path in split_dic['train']['labels']:
        train_classes += read_label(label_path).sum(axis=(1, 2)).astype(np.int64)

    # Find the classes present in the testing set
    for label_path in split_dic['test']['labels']:
        test_classes += read_label(label_path).sum(axis=(1, 2)).astype(np.int64)

    # All the classes in the testing set must be present in the training set
    for idx, count in enumerate(test_classes):
        if count > 0 and train_classes[idx] == 0:
            return False
    
    return True


def copy_into_dir(file_list, dst_dir):
    """
    @brief Copies the files in the list into the provided destination directory.
    @param[in]  file_list  List of paths to files.
    @param[in]  dst_dir    Path to the destination directory.
    @returns nothing.
    """
    for src_path in file_list:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename) 
        shutil.copyfile(src_path, dst_path)


def copy_data(split_dic, train_path, test_path):
    """
    @brief Gets a split of training/testing files and copies them into separate
           folders.
    @param[in]  split_dic  Dictionary of dictionaries. Keys 'train' and 'test' 
                           in the upper level. Keys 'images' and 'labels' in 
                           the lower level.
    @returns nothing.
    """
    copy_into_dir(split_dic['train']['images'], train_path) 
    copy_into_dir(split_dic['train']['labels'], train_path) 
    copy_into_dir(split_dic['test']['images'], test_path) 
    copy_into_dir(split_dic['test']['labels'], test_path) 


def compute_dataset_stats(images, labels):
    # Gather data for report
    no_classes = len(classnames)
    no_pixels = np.zeros((no_classes,), dtype=np.int64)
    no_images = np.zeros((no_classes,), dtype=np.int64)
    
    # Find the number of pixels and images per class
    for label_path in labels:
        label = read_label(label_path).sum(axis=(1, 2)).astype(np.int64)
        no_pixels += label
        label[label > 0] = 1
        no_images += label 

    return no_pixels, no_images


def print_dataset_stats(no_pixels, no_images, labels): 
    """
    @brief Prints the number of pixels and images per class.
    @param[in]  no_pixels  Number of pixels of each class, np.ndarray, shape (C,).
    @param[in]  no_images  Number of images of each class, np.ndarray, shape (C,).
    @returns nothing.
    """
    no_classes = len(classnames)
    print('Number of classes:', no_classes)
    print('Number of annotated images in the dataset:', len(labels))
    print('Number of pixels per class:')
    no_classes = len(classnames)
    margin = '                      '
    for c in range(no_classes):
        print('   ' + str(classnames[c]) + ' [' + str(c) + ']:' \
            + margin[:len(margin) - len(classnames[c]) - len(str(c))] \
            + str(no_pixels[c]))
    print('Number of images per class:')
    for c in range(no_classes):
        print('   ' + str(classnames[c]) + ' [' + str(c) + ']:' \
            + margin[:len(margin) - len(classnames[c]) - len(str(c))] \
            + str(no_images[c]))


def singular_classes(no_images):
    """
    @brief Get a list of classes that are present in only one image of the 
           dataset.
    @param[in]  no_images  Number of images per class, shape (C,).
    @returns a list of the class indices. 
    """
    return [c for c in range(no_images.shape[0]) if no_images[c] < 2]


def save_dataset_metadata(images, labels, path):
    """
    @brief Save JSON file with the information of the dataset.
    @param[in]  images  List of image paths.
    @param[in]  labels  List of label image paths.
    @param[in]  path    String with the path of the dataset folder.
    @returns nothing.
    """
    data = {}

    # Append class indices
    idx2class = classnames
    class2idx = {y: x for x, y in idx2class.items()}
    data['idx2class'] = idx2class 
    data['class2idx'] = class2idx

    # Append a list of the images that should be found in the dataset
    data['images'] = images

    # Append a list of the labels that should be found in the dataset
    data['labels'] = labels

    # Append the number of images
    data['number_of_images'] = len(images) 

    # Append the list of images per class and the number of pixels per class
    data['images_per_class'] = {c: [] for c in class2idx}
    data['number_pixels_per_class'] = {c: 0 for c in class2idx}
    for image_path, label_path in zip(images, labels):
        label = read_label(label_path).sum(axis=(1, 2)).astype(np.int64)
        for class_name in data['images_per_class']:
            class_id = class2idx[class_name]  
            if label[class_id] > 0: 
                data['images_per_class'][class_name].append(image_path)
            data['number_pixels_per_class'][class_name] += int(label[class_id])

    # Append the number of images per class
    data['number_images_per_class'] = {c: len(data['images_per_class'][c]) \
        for c in class2idx}

    # Append ignore labels, that is, those classes that are not present
    # in the dataset
    data['ignore_labels'] = [c for c in data['number_images_per_class'] \
        if data['number_images_per_class'][c] == 0]
    
    # Create metadata file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Create output folder
    os.mkdir(args.output)

    # Get list of image and segmentation files
    images, labels = read_files(args.input)
    no_pixels, no_images = compute_dataset_stats(images, labels)
    #print_dataset_stats(no_pixels, no_images, labels)

    # Get the list of singular classes, i.e. those that are only present in one
    # image, we will put these images in the training set
    #sc = [classnames[c] for c in singular_classes(no_images)]
    #print('Classes that are only in one image:', sc)

    # TODO: Save dataset metadata
    dataset_metadata_path = os.path.join(args.output, 'metadata.txt')
    save_dataset_metadata(images, labels, dataset_metadata_path)

    # Generate the split for each fold
    for i in tqdm.tqdm(range(args.folds)):
        # Get a valid split of the data
        valid_split = False 
        split_dic = None
        attempt = 0
        while not valid_split:
            attempt += 1
            sys.stdout.write('Attempt to generate a valid split... ')
            sys.stdout.flush()
            split_dic = generate_random_split(images, labels, args.train)
            valid_split = validate_random_split(split_dic)
            if valid_split:
                sys.stdout.write("[OK]\n")
            else:
                sys.stdout.write("[FAIL]\n")

        # Create output folders
        sys.stdout.write('Creating output folders... ')
        sys.stdout.flush()
        fold_path = os.path.join(args.output, 'fold_' + str(i))
        fold_train_path = os.path.join(fold_path, 'train')
        fold_test_path = os.path.join(fold_path, 'test')
        os.mkdir(fold_path)
        os.mkdir(fold_train_path)
        os.mkdir(fold_test_path)
        sys.stdout.write("[OK]\n")

        # Save fold metadata
        train_metadata_path = os.path.join(fold_path, 'train_metadata.txt')
        test_metadata_path = os.path.join(fold_path, 'test_metadata.txt')
        save_dataset_metadata(split_dic['train']['images'],
            split_dic['train']['labels'], train_metadata_path)
        save_dataset_metadata(split_dic['test']['images'],
            split_dic['test']['labels'], test_metadata_path)

        # Copy data to the fold 
        sys.stdout.write('Copying data to the fold... ')
        sys.stdout.flush()
        copy_data(split_dic, fold_train_path, fold_test_path)
        sys.stdout.write("[OK]\n")

if __name__ == '__main__':
    main()
