"""
@brief  Module that contains loss functions.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021. 
"""

import numpy as np
import sys
import torch
import monai.losses


def nll_loss(pred, gt):
    """
    @brief Negative log-likelihood loss (also called multi-class cross entropy). 
    @param[in]  pred  (N,C) where C = number of classes.
    @param[in]  gt    (N) where each value is 0 <= targets[i] <= C - 1.
    @returns a scalar with the mean loss.
    """ 
    return torch.nn.functional.nll_loss(pred, gt)


def focal_loss(pred, gt):
    """
    @param[in]  pred  Tensor of probability predictions, shape (B, C, H, W).
    @param[in]  gt    Tensor of ground truth, shape (B, C, H, W).
    """
    # FIXME: not working
    raise NotImplemented()
    fl = monai.losses.FocalLoss()
    return fl(pred, gt)


def odsi_db_pw_ce_logprob_loss(raw_pred, raw_gt):
    """
    @brief Pixel-wise cross entropy loss.
    @details Reduces the loss for the batch using the mean.
    @param[in]  raw_pred  Tensor of predicted log-probabilities, 
                          shape (B, C, H, W).
    @param[in]  raw_gt    Tensor of ground truth probabilities, 
                          shape (B, C, H, W).
    @returns a scalar with the mean loss over the images of the batch.
    """

    # Class weights according to the number of pixels per class in ODSI-DB,
    # a higher number indicates less represented classes
    class_weights = torch.cuda.FloatTensor([
        0.9682632484506389, 0.9983340617620862, 0.9999394663490677, 
        0.9995276328275359, 0.9998907852100988, 0.9188844950588891, 
        0.9999902109476404, 0.9997049747997107, 0.9973278363209698, 
        0.97715387049886,   0.9607777928694627, 0.9986612604245154, 
        0.9996366990483456, 0.9999236850100192, 0.9352800022239143, 
        0.9999932978832073, 0.9999784739894149, 0.986721340312427, 
        0.9967532371059031, 0.9997563631976796, 0.99995392707397, 
        0.8692997588756661, 0.860435994817646,  0.9992877927908862, 
        0.9998345270474653, 0.9957902617780788, 0.9875275587410823, 
        0.9997695198161124, 0.9879129304566572, 0.6864583697190043, 
        0.9769124622329116, 0.9681097270561285, 0.9996792888545647, 
        0.9326207970711948, 0.9999083493782449])

    # Flatten predictions and labels
    bs = raw_pred.shape[0]
    chan = raw_pred.shape[1] 
    pred_log = torch.reshape(raw_pred, (bs, chan, -1))
    gt = torch.reshape(raw_gt, (bs, chan, -1))
    #npixels = pred_log.shape[2]
    
    # Loop over the batch (each image might have different number of annotated
    # pixels)
    ce_loss = torch.empty((bs))
    for i in range(bs):
        # Filter out those pixels without labels
        ann_idx = torch.sum(gt[i, ...], dim=0) == 1
        num_valid_pixels = ann_idx.float().sum()

        # Convert prediction into nll_loss() preferred input shape (N, C)
        log_yhat = pred_log[i, :, ann_idx].permute(1, 0)

        # Convert vector of probabilities into target vector
        y = torch.argmax(gt[i, :, ann_idx], dim=0)

        # Compute pixel-wise multi-class (exclusive) cross-entropy loss 
        ce_loss[i] = torch.nn.functional.nll_loss(log_yhat, y, 
                                                  weight=class_weights)

    # NOTE: The cross entropy for some of the images might be nan, this happens
    # when an image does not have properly annotated pixels

    # Reduce the loss for the batch using the mean
    is_nan = torch.isnan(ce_loss)
    ce_loss[is_nan] = 0
    loss = ce_loss.sum() / (~is_nan).float().sum()

    return loss


def ResNet_18_CAM_DS_ce_loss(tuple_pred, raw_gt):
    """
    @brief  Computes the cross entropy loss for all the images in the batch
            (taking into account the predictions at all resolution levels).

    @details  This loss estimates how good is the network at estimating
              which classes are present in a given image.

              We treat this problem as many binary classification problems,
              each class versus all the others.

    @param[in]  tuple_pred  Tuple of six elements, the first one is a tensor 
                            containing the predicted class presence 
                            log-probabilities, shape (R, B, K, 1, 1).

                            * B is the batch size.

                            * R will be equivalent to the number of resolution 
                              levels + 1 (the global prediction). 

                            * K is the number of classes. 

                            The rest of the elements of the tuple are the
                            pseudo-segmentation predictions at different 
                            resolutions.
                            
    @param[in]  raw_gt      Tensor of ground truth probabilities, 
                            shape (B, C, H, W).
    """

    # Get the classification predictions, we do not use the
    # pseudo-segmentations to calculate the loss
    raw_pred = tuple_pred[0]
    
    # Get dimensions
    bs = raw_pred.shape[1]
    classes = raw_gt.shape[1]
    rl = raw_pred.shape[0]  # Resolution levels + 1 (the global loss)
    
    # Loop over the images of the batch
    ce_loss = torch.cuda.FloatTensor(bs).fill_(0)
    for i in range(bs):
        # Build ground truth vector of class-presence probabilities from the 
        # segmentation ground truth that ODSI-DB provides
        gt = torch.cuda.FloatTensor(classes).fill_(0)
        for j in range(classes):
            gt[j] = 1 if 1 in raw_gt[i, j] else 0

        # Compute the losses at all resolution levels + global
        for j in range(rl):
            # Build vector of class-presence log-probability predictions
            pred = torch.flatten(raw_pred[j, i])

            # Compute loss at this resolution level and add it to the accumulator
            ce_loss_per_class = - (gt * pred + (1 - gt) * torch.log(1 - torch.exp(pred)))

            # Mean over classes
            ce_loss[i] += ce_loss_per_class.mean()

    # Reduce the loss for the batch using the mean
    is_nan = torch.isnan(ce_loss)
    ce_loss[is_nan] = 0
    loss = ce_loss.sum() / (~is_nan).float().sum()

    return loss


if __name__ == '__main__':
    raise RuntimeError('The loss.py module is not is not a script.') 
