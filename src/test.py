"""
@brief  Main script to launch the testing process.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   5 Jun 2021.
"""

import argparse
import collections
import torch
import numpy as np
import tqdm
import json
import cv2
import skimage.color

# My imports
import torchseg.config.parser
import torchseg.data_loader as dl
import torchseg.model
import torchseg.machine


# There are 35 classes plus background in ODSI-DB
palette = np.round(np.array([
       [0.0       , 0.0,        0.0       ],
       [0.73910129, 0.54796227, 0.70659469],
       [0.07401779, 0.48485457, 0.2241555 ],
       [0.35201324, 0.9025658 , 0.81062183],
       [0.08126211, 0.23986311, 0.54880697],
       [0.33267484, 0.6119932 , 0.30272535],
       [0.45419585, 0.1818727 , 0.5877175 ],
       [0.1239585 , 0.17862775, 0.6892662 ],
       [0.20556493, 0.44462774, 0.38364081],
       [0.18754881, 0.1831789 , 0.00863592],
       [0.37702173, 0.075744  , 0.07170247],
       [0.9487006 , 0.90159635, 0.26639963],
       [0.8954375 , 0.58731839, 0.87918311],
       [0.83980577, 0.77131811, 0.02192928],
       [0.47681103, 0.72962211, 0.96439166],
       [0.44293943, 0.60166042, 0.5879358 ],
       [0.52419707, 0.18690438, 0.69027514],
       [0.34720014, 0.57450984, 0.96570434],
       [0.78380941, 0.2237716 , 0.52199938],
       [0.98170786, 0.61735585, 0.73834123],
       [0.44000012, 0.06259595, 0.76726459],
       [0.47754739, 0.13137904, 0.04615173],
       [0.65486219, 0.24028978, 0.75424866],
       [0.79301129, 0.75970907, 0.06562084],
       [0.14864707, 0.55623561, 0.80328385],
       [0.54439947, 0.234355  , 0.81248573],
       [0.24443958, 0.00697174, 0.59921356],
       [0.76808718, 0.56387681, 0.52199431],
       [0.69855907, 0.73646473, 0.8320837 ],
       [0.85436454, 0.86456808, 0.61494475],
       [0.34944949, 0.79188401, 0.8251793 ],
       [0.43554137, 0.18054355, 0.80210866],
       [0.76501493, 0.38795293, 0.49637574],
       [0.31552006, 0.3704537 , 0.90083695],
       [0.26176471, 0.66781917, 0.65375891],
       [0.21141543, 0.16505171, 0.53799316],
]) * 255.).astype(np.uint8)


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-c': 'config file path (default: None)',
        '-l': 'logger config file path (default: None)',
        '-r': 'path to latest checkpoint (default: None)',
        '-d': 'indices of GPUs to enable (default: all)',
        #'-o': 'path to the output JSON file (default: None)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch segmenter.')
    args.add_argument('-c', '--conf', default=None, type=str, help=help('-c'))
    args.add_argument('-l', '--logconf', default=None, type=str, help=help('-l'))
    args.add_argument('-r', '--resume', default=None, type=str, help=help('-r'))
    args.add_argument('-d', '--device', default=None, type=str, help=help('-d'))
    #args.add_argument('-o', '--output', default=None, type=str, help=help('-o'))
    return args


def parse_config(args):
    """
    @brief Combines parameters from both JSON and command line into a 
    @param[in]  args  Argparse args object.
    @returns A torchseg.config.parser.ConfigParser object.
    """
    # Custom CLI options to modify the values provided in the JSON configuration
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-o', '--output'], type=str, target='output'),
    ]
    config = torchseg.config.parser.ConfigParser.from_args(args, options)
    return config


def image_based_accuracy(log_pred, gt, nclasses=35):
    # Convert log-probabilities into probabilities
    pred = torch.exp(log_pred)

    # Remove the batch dimension   
    pred = pred[0, :]
    gt = gt[0, :]

    # Flatten the images
    pred = torch.reshape(pred, (nclasses, -1))
    gt = torch.reshape(gt, (nclasses, -1)) 

    # List of classes considered
    idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames
    class2idx = {y: x for x, y in idx2class.items()}
    relevant_class = ["Skin", "Oral mucosa", "Enamel", "Tongue", "Lip", "Hard palate", 
                      "Attached gingiva", "Soft palate", "Hair"]
    relevant_class_idx = torch.tensor([class2idx[x] for x in relevant_class])

    # Get only the annotated pixels
    ann_idx = torch.sum(gt, dim=0) == 1
    y_pred = pred[:, ann_idx]
    y_true = gt[:, ann_idx]

    # Discard non-relevant classes
    y_pred = y_pred[relevant_class_idx, :]
    y_true = y_true[relevant_class_idx, :]

    # Get class index predictions 
    y_pred = torch.argmax(y_pred, dim=0)
    y_true = torch.argmax(y_true, dim=0)
    
    # Compute accuracy
    correct_predictions = (y_pred == y_true).float().sum()
    total_predictions = y_true.shape[0]
    accuracy = (correct_predictions / total_predictions).item()
    
    return accuracy


def patch_based_prediction(data, gt, model, device, patch_size=512):
    # Create full prediction
    output = torch.zeros_like(gt).to(device)

    im_height = data.shape[2]
    im_width = data.shape[3]
    
    i = 0
    j = 0
    while j < im_height:
        # Compute patch height
        j_end = j + patch_size
        if j_end > im_height:
            j_end = im_height

        while i < im_width:
            # Compute patch width
            i_end = i + patch_size
            if i_end > im_width:
                i_end = im_width

            # Get patch into GPU
            patch = data[:, :, j:j_end, i:i_end]
            patch = patch.to(device)

            # Perform patch inference
            output[:, :, j:j_end, i:i_end] = model(patch)
            
            # Move to the right
            i += patch_size

        # Move to the bottom
        j += patch_size
    
    return output




def label2bgr(im, pred, gt):
    """
    @brief Function to convert a segmentation label into an RGB image.
           Label is expected to be a Torch tensor with shape C, H, W.
    """

    # Convert prediction and ground truth to single channel
    pred_sinchan = torch.argmax(pred, dim=0).numpy()
    gt_sinchan = torch.argmax(gt, dim=0).numpy()

    # Add one to all the classes because the non-annotated pixels will be black 
    pred_sinchan += 1
    gt_sinchan += 1
    
    # Black out the non-annotated pixels
    nan_idx = torch.sum(gt, dim=0) != 1
    pred_sinchan[nan_idx, ...] = 0
    gt_sinchan[nan_idx, ...] = 0

    # List of classes considered
    idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames
    class2idx = {y: x for x, y in idx2class.items()}
    relevant_class = ["Skin", "Oral mucosa", "Enamel", "Tongue", "Lip", "Hard palate", 
                      "Attached gingiva", "Soft palate", "Hair"]
    relevant_class_idx = [class2idx[x] for x in relevant_class]

    # Black out the non-relevant classes
    for idx in relevant_class_idx:
        pixels_of_this_class = gt_sinchan == (idx + 1)
        pred_sinchan[pixels_of_this_class] = 0
        gt_sinchan[pixels_of_this_class] = 0

    # Convert single-channel label prediction and ground truth to BGR
    pred_bgr = skimage.color.label2rgb(pred_sinchan, colors=palette)
    gt_bgr = skimage.color.label2rgb(gt_sinchan, colors=palette) 

    return pred_bgr, gt_bgr


def main():
    # Read command line arguments
    args = parse_cmdline_params()

    # Read configuration file
    config = parse_config(args)
    logger = config.get_logger('test')

    # Hack config file to add 'odsi_db_conf_mat' as a metric
    config['metrics'].append('odsi_db_conf_mat')

    # Get testing logger
    logger = config.get_logger('test')

    # Build network architecture
    model = config.init_obj('model', torchseg.model)
    logger.info(model)
    
    # Load the weights of a trained architecture (typically model_best.pth)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if torch.cuda.is_available() and config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict) 
    model = model.to(device)

    # Put model in test mode
    model.eval()

    # Get function handles of loss and metrics
    loss_fn = getattr(torchseg.model.loss, config['loss'])
    metric_fns = [getattr(torchseg.model.metric, met) \
        for met in config['metrics']]

    # Loop over all testing datasets
    for dataset in config['testing']['datasets']:
        # Create dataloader for this particular testing dataset
        data_loader = getattr(torchseg.data_loader, 
            dataset['type'])(**dataset['args'])
        data_loader.training = False

        # Initialise image-based accuracy accumulator
        im_data = []
        im_pred = []
        im_gt = []
        im_acc = []
        im_path = []

        # Run inference on all the samples in the dataset
        total_loss = 0.0
        #total_metrics = torch.zeros(len(metric_fns))
        total_metrics = [None] * len(metric_fns)
        counter = 0
        with torch.no_grad():
            for i, raw_data in enumerate(tqdm.tqdm(data_loader)):
                counter += 1

                # Load input data
                data = raw_data['image']
                gt = raw_data['label']
                path = raw_data['path']
                
                # Run inference on the GPU
                output = None
                if data.shape[3] > 512 and data_loader.mode in ['rgbpixel_test', 'spixel_170_test']:
                    output = patch_based_prediction(data, gt, model, device)
                    gt = gt.to(device)
                else:
                    # Send image and label to GPU
                    data, gt = data.to(device), gt.to(device)

                    # Perform inference on the input
                    output = model(data)
                
                # Update image-based accuracy
                im_data.append(data.detach().cpu()[0, ...])
                im_pred.append(output.detach().cpu()[0, ...])
                im_gt.append(gt.detach().cpu()[0, ...])
                im_acc.append(image_based_accuracy(output.detach().cpu(), gt.detach().cpu()))
                im_path.append(path)

                # Update accumulated loss
                loss = loss_fn(output, gt)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                # Update accumulated metrics
                for i, metric in enumerate(metric_fns):
                    metric_value = metric(output, gt)
                    if total_metrics[i] is None:
                        total_metrics[i] = metric_value * batch_size
                    else:
                        total_metrics[i] += metric_value * batch_size

        # Get best, median, worst image index
        best_idx = np.argmax(im_acc) 
        median_idx = np.argsort(im_acc)[len(im_acc) // 2]
        worst_idx = np.argmin(im_acc) 
        
        # FIXME: debugging
        pred_bgr, gt_bgr = label2bgr(im_data[best_idx], im_pred[best_idx], im_gt[best_idx])
        cv2.imwrite('/tmp/pred_bgr.png', pred_bgr)
        cv2.imwrite('/tmp/gt_bgr.png', gt_bgr)

        # Calculate the final image-based accuracy
        print('Image-based minimum accuracy:', im_acc[worst_idx])
        print('Image-based median accuracy:', im_acc[median_idx])
        print('Image-based maximum accuracy:', im_acc[best_idx])
        print('Image-based average accuracy:', sum(im_acc) / len(im_acc))
        
        #print('Batch size:', batch_size)
        #print('Metric functions:', metric_fns)
        #print('Total metrics:', [x / counter for x in total_metrics])

        # Find the index of the 'confusion matrix' metric
        conf_mat_idx = None
        for i, met in enumerate(metric_fns):
            if met.__name__ == 'odsi_db_conf_mat':
                conf_mat_idx = i
        if not conf_mat_idx:
            raise RuntimeError('Confusion matrix fake metric not found.')

        # Get confusion matrix (accumulated over all the testing pixels)
        conf_mat = total_metrics[conf_mat_idx][0].detach().cpu().numpy()

        # Compute the balance accuracy per class
        bal_acc = {}
        sensitivity = {}  
        specificity = {}
        accuracy = {}
        i2c = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames 
        for i in range(conf_mat.shape[0]):
            tp, fp, tn, fn = conf_mat[i]
            sensitivity[i2c[i]] = 1. * tp / (tp + fn) 
            specificity[i2c[i]] = 1. * tn / (tn + fp)
            accuracy[i2c[i]] = 1. * (tp + tn) / (tp + tn + fp + fn)
            bal_acc[i2c[i]] = .5 * sensitivity[i2c[i]] + .5 * specificity[i2c[i]]

        # Save JSON file with the results per class
        with open(config['output'] + '_sensitivity.json', 'w') as f:
            json.dump(sensitivity, f)
        with open(config['output'] + '_specificity.json', 'w') as f:
            json.dump(specificity, f)
        with open(config['output'] + '_accuracy.json', 'w') as f:
            json.dump(accuracy, f)
        with open(config['output'] + '_balanced_accuracy.json', 'w') as f:
            json.dump(bal_acc, f)
        
        # Create a dictionary with the resuls on this dataset
        #n_samples = len(data_loader.sampler)
        #log = {'loss': total_loss / n_samples}
        #log.update({
        #    met.__name__: total_metrics[i].item() / n_samples \
        #        for i, met in enumerate(metric_fns)
        #})

        # Show or save the results for this dataset in the log
        #logger.info({dataset['type']: log})


if __name__ == '__main__':
    main()
