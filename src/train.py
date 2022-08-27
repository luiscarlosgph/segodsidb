"""
@brief  Main script to kick off the training. 
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021.
"""

import argparse
import collections
import torch
import numpy as np

# My imports
import torchseg.config.parser
import torchseg.data_loader
import torchseg.model
import torchseg.machine

# Fix random seeds for reproducibility
SEED = 18303
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-c': 'config file path (default: None)',
        '-l': 'logger config file path (default: None)',
        '-r': 'path to latest checkpoint (default: None)',
        '-d': 'indices of GPUs to enable (default: all)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch segmenter.')
    args.add_argument('-c', '--conf', default=None, type=str, help=help('-c'))
    args.add_argument('-l', '--logconf', default=None, type=str, help=help('-l'))
    args.add_argument('-r', '--resume', default=None, type=str, help=help('-r'))
    args.add_argument('-d', '--device', default=None, type=str, help=help('-d'))
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
        CustomArgs(['--lr', '--learning-rate'], type=float, 
            target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch-size'], type=int, 
            target='data_loader;args;batch_size'),
        CustomArgs(['--data-dir'], type=str, 
            target='data_loader;args;data_dir'),
        CustomArgs(['--save-dir'], type=str, 
            target='machine;args;save_dir'),
    ]
    config = torchseg.config.parser.ConfigParser.from_args(args, options)
    return config


def main():
    args = parse_cmdline_params()
    config = parse_config(args)

    # Note: the 'type' in config.json indicates the class, and the 'args' in 
    # config.json will be passed as parameters to the constructor of that class
    
    # Get training logger
    logger = config.get_logger('train')

    # Setup data loader
    data_loader = config.init_obj('data_loader', torchseg.data_loader)
    valid_data_loader = data_loader.split_validation()

    # Create model 
    model = config.init_obj('model', torchseg.model)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = torchseg.utils.setup_gpu_devices(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and evaluation metrics
    criterion = getattr(torchseg.model.loss, config['loss'])
    metrics = [getattr(torchseg.model.metric, m) for m in config['metrics']]

    # Create optmizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # Learning rate scheduler
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, 
        optimizer)

    # Create learning machine
    trainer = torchseg.machine.GenericMachine(model, criterion, metrics, 
                                              optimizer,
                                              config=config,
                                              device=device,
                                              data_loader=data_loader,
                                              valid_data_loader=valid_data_loader,
                                              lr_scheduler=lr_scheduler,
                                              acc_steps=config['machine']['args']['acc_steps'])
    # TODO: We should be able to create the learning machine just with this type 
    #       of call
    #trainer = config.init_obj('machine', torchseg.machine)
    
    # Launch training
    trainer.train()


if __name__ == '__main__':
    main()
