"""
@brief  Module that contains the learning machine for each problem/task.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   2 Jun 2021.
"""

import numpy as np
import torch
import torchvision.utils

# My imports
import torchseg.base
import torchseg.utils


class GenericMachine(torchseg.base.BaseMachine):
    """
    @class Generic learning machine.
    @param[in]  model              TODO
    @param[in]  criterion          TODO
    @param[in]  metric_ftns        TODO
    @param[in]  optimizer          TODO
    @param[in]  config             TODO
    @param[in]  device             TODO
    @param[in]  data_loader        TODO
    @param[in]  valid_data_loader  TODO
    @param[in]  lr_scheduler       TODO
    @param[in]  len_epoch          TODO
    @param[in]  acc_steps          Number of batches used to accumulate gradients.
                                   Increasing this value is (almost) equivalent to increasing 
                                   the batch size, but without using more GPU memory. Obviously,
                                   this will slow down the training.
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, 
                 len_epoch=None, acc_steps=1):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # Epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # Iteration-based training
            self.data_loader = torchseg.utils.inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.acc_steps = acc_steps
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = torchseg.utils.MetricTracker('loss', 
            *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = torchseg.utils.MetricTracker('loss', 
            *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        @brief Training logic for an epoch.
        @param[in]  epoch  Integer, current training epoch.
        @returns a log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.data_loader.training = True
        
        # Data structures for accumulated gradients
        acc_batches = 0
        acc_loss = 0
        self.optimizer.zero_grad()

        # For all the batches of this epoch
        for batch_idx, raw_data in enumerate(self.data_loader):
            # Machinery for accumulated gradients, which can be used by users when they want to
            # use a larger batch size than what fits in GPU
            acc_batches += 1

            # Load batch (and expected output) into the GPU, len(data) here must be equal
            # to your batch size
            data = raw_data['image']
            target = raw_data['label']
            data, target = data.to(self.device), target.to(self.device)

            #import cv2
            #foo = raw_data['image'][0].numpy().transpose((1, 2, 0)).astype(
            #    np.uint8)[...,::-1]
            #print('Foo:', foo.shape)
            #cv2.imshow('image', foo)
            #fool = raw_data['label'][0, 0, :, :].numpy() * 255
            #cv2.imshow('label', fool)
            #cv2.waitKey(0)
            
            # Perform inference
            output = self.model(data)

            # Compute loss
            loss = self.criterion(output, target) / self.acc_steps

            # Update weights with backpropagation 
            loss.backward()
            acc_loss += loss.item()

            # Machinery for accumulated gradients
            if acc_batches == self.acc_steps: 
                self.logger.debug('Train Epoch: {} {} LR: {} Accumulated loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    [ group['lr'] for group in self.optimizer.param_groups ],
                    acc_loss))
                self.optimizer.step()
                acc_batches = 0
                acc_loss = 0
                self.optimizer.zero_grad()

             

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item() * self.acc_steps)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} LR: {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    [ group['lr'] for group in self.optimizer.param_groups ],
                    loss.item() * self.acc_steps))
                # FIXME: Add image only if it has three channels
                #self.writer.add_image('input', 
                #    torchvision.utils.make_grid(data.cpu(), nrow=8,
                #    normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if type(self.lr_scheduler) == \
                    torch.optim.lr_scheduler.ReduceLROnPlateau:
                #self.lr_scheduler.step(self.train_metrics.avg(self.mnt_metric))
                self.lr_scheduler.step(val_log['loss'])
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        @brief Validate after training an epoch.
        @param[in]  epoch  Integer, current training epoch.
        @returns A log that with the validation information.
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.data_loader.training = False 

        with torch.no_grad():
            #for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            for batch_idx, raw_data in enumerate(self.data_loader):
                data = raw_data['image'] 
                target = raw_data['label'] 
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)

                print('Validation epoch:', epoch - 1)
                print('len(self.valid_data_loader):', len(self.valid_data_loader))
                print('batch_idx:', batch_idx)

                self.writer.set_step((epoch - 1) \
                    * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                # FIXME: Add image only if it has three channels
                #self.writer.add_image('input', 
                #    torchvision.utils.make_grid(data.cpu(), nrow=8, 
                #    normalize=True))

        # Add histogram of model parameters to the tensorboard
        # FIXME: disabled as it is not currently used
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """
        @brief TODO
        @param[in]  batch_idx  TODO
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


