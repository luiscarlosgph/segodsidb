"""
@brief  Module with base learning machine. This class is meant to control the
        training and validation processes for a particular problem or task.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   2 Jun 2021.
"""

import torch
import abc
import numpy as np

# My imports
import torchseg.visualization


class BaseMachine:
    """
    @class BaseMachine is an interface class, parent of for all the convolutional
           learning machines.
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('machine', config['machine']['args']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_machine = config['machine']['args']
        self.epochs = cfg_machine['epochs']
        self.save_period = cfg_machine['save_period']
        self.monitor = cfg_machine.get('monitor', 'off')

        # Configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = cfg_machine.get('early_stop', np.inf)
            if self.early_stop <= 0:
                self.early_stop = np.inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # Setup visualization writer instance                
        self.writer = torchseg.visualization.TensorboardWriter(config.log_dir, 
            self.logger, cfg_machine['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)
    
    @abc.abstractmethod
    def _train_epoch(self, epoch):
        """
        @brief Training logic for an epoch.
        @param[in]  epoch  Current epoch number.
        """
        raise NotImplementedError

    def train(self):
        """
        @brief Full training logic.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # Print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # Evaluate model performance according to configured metric, 
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # Check whether model performance improved or not, 
                    # according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' \
                        and log[self.mnt_metric] <= self.mnt_best) \
                        or (self.mnt_mode == 'max' \
                        and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("""Warning: Metric '{}' is not found. 
                                           Model performance monitoring is 
                                           disabled.""".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info('Validation performance did not improve ' \
                                     + 'for ' + str(self.early_stop)           \
                                     + ' epochs. Training stops.')
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        @brief Method to save checkpoints.

        @param[in]  epoch      Current epoch number.
        @param[in]  log        Logging information of the epoch.
        @param[in]  save_best  If True, rename the saved checkpoint to 
                               'model_best.pth'.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(
            epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        @brief Resume from saved checkpoints.
        @param[in]  resume_path  Checkpoint path to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # Load architecture params from checkpoint
        #if checkpoint['config']['arch'] != self.config['arch']:
        #    self.logger.warning('Warning: Architecture configuration given in \
        #                         config file is different from that of \
        #                         checkpoint. This may yield an exception while \
        #                         state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Load optimizer state from checkpoint only when optimizer type is not 
        # changed
        if checkpoint['config']['optimizer']['type'] != \
                self.config['optimizer']['type']:
            self.logger.warning("""Warning: Optimizer type given in config file is
                                   different from that of checkpoint.
                                   Optimizer parameters not being resumed.""")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('Checkpoint loaded. Resuming training from epoch ' \
                         + str(self.start_epoch))



