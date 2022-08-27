"""
@brief  Collection of deep learning models. 
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   4 Jun 2021.
"""

import torch
import torch.nn.functional as F
import torch.utils.model_zoo
import numpy as np
import monai.networks
import torchvision.models
import collections
import math
import unet

# My imports
import torchseg.base


# PyTorch pretrained model URLs
model_urls = {
    "deeplabv3_resnet50_coco": 
        "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": 
        "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_mobilenet_v3_large_coco": 
        "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
}


class MnistModel(torchseg.base.BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleUnet(torchseg.base.BaseModel):
    """
    @brief Cut-down implementation of a small U-Net. This network is here for
           learning and debugging purposes.
    @details Four resolution levels with 32, 64, 128, and 128 neurons. 
    """

    def __init__(self, in_channels=3, out_channels=35):
        """
        @param[in]  in_channels  Number of input channels, e.g. set it to
                                        three for RGB images.
        @param[in]  out_channels Number of classes.
        """
        super().__init__()
        
        # Way down
        self.conv1 = self.down_block(in_channels, 32, 7, 3)
        self.conv2 = self.down_block(32, 64, 3, 1)
        self.conv3 = self.down_block(64, 128, 3, 1)
        
        # Way up
        self.upconv3 = self.up_block(128, 64, 3, 1)
        self.upconv2 = self.up_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.up_block(32 * 2, out_channels, 3, 1)

    def down_block(self, in_channels, out_channels, kernel_size, padding):
        layer = torch.nn.Sequential(
            # First convolution
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            
            # Second convolution
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            
            # Max-pooling
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return layer

    def up_block(self, in_channels, out_channels, kernel_size, padding):
        layer = torch.nn.Sequential(
            # First convolution 
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            # Second convolution
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, 
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            # Up-convolution
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, 
                stride=2, padding=1, output_padding=1) 
        ) 
        return layer

    def crop(self, x, ref):
        if x.shape != ref.shape:
            crop_h = ref.shape[2] - x.shape[2] 
            crop_up = int(np.floor(crop_h / 2))
            crop_down = crop_h - crop_up
            crop_w = ref.shape[3] - x.shape[3] 
            crop_left = int(np.floor(crop_w / 2))
            crop_right = crop_w - crop_left
            x = F.pad(x, (crop_left, crop_right, crop_up, crop_down))
        return x

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        upconv3 = self.upconv3(conv3)
        upconv3 = self.crop(upconv3, conv2)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv2 = self.crop(upconv2, conv1)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        upconv1 = self.crop(upconv1, x)

        return F.log_softmax(upconv1, dim=1)


class Unet(torchseg.base.BaseModel):
    """
    @class Wraps the 'pip install unet' U-Net implementation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 35) -> None:
        """
        @param[in]  in_channels   Number of channels of the input image.
        @param[in]  out_channels  Number of output classes.
        """
        super().__init__()

        # Get segmentation model
        self.backbone = unet.UNet(in_channels=in_channels,
                                  out_classes=out_channels,
                                  dimensions=2,
                                  num_encoding_blocks=5,  
                                  out_channels_first_layer=64,
                                  normalization='batch',
                                  padding=1,
                                  preactivation=True,
                                  residual=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Get input shape
        bs, channels, old_height, old_width = x.shape

        # Compute form factor
        ff = float(old_width) / old_height

        # Resize width to 512
        width = 512
        height = int(round(width / ff))
        x = torchvision.transforms.functional.resize(x, [height, width])
        
        # Find how much we have to pad width and height
        powers = [2**i for i in range(9, 13)]
        height_padding = pow(2, math.ceil(np.log2(height))) - height
        width_padding = pow(2, math.ceil(np.log2(width))) - width

        # Add padding to the image
        x = F.pad(x, (0, width_padding, 0, height_padding), 'constant', 0)

        # Perform forward pass with unet.UNet 
        x = self.backbone(x)

        # Remove padding and convet the image to the original size
        x = F.pad(x, (0, -width_padding, 0, -height_padding), 'constant', 0)

        # Resize back to original size
        x = torchvision.transforms.functional.resize(x, [old_height, old_width])

        # Return log-probabilities
        return F.log_softmax(x, dim=1)


class Unet_obsolete(torchseg.base.BaseModel):
    """
    @brief Classical U-Net implementation.
    """

    def __init__(self, in_channels=3, out_channels=35):
        """
        @param[in]  in_channels  Number of input channels, e.g. set it to
                                 three for RGB images.
        @param[in]  out_channels Number of classes.
        """
        super().__init__()
        
        # Way down
        self.conv1 = self.down_block(in_channels, 64, 7, 3)
        self.conv2 = self.down_block(64, 128, 3, 1)
        self.conv3 = self.down_block(128, 256, 3, 1)
        self.conv4 = self.down_block(256, 512, 3, 1)
        self.conv5 = self.down_block(512, 1024, 3, 1)
        
        # Way up
        self.upconv5 = self.up_block(1024, 512, 3, 1)
        self.upconv4 = self.up_block(512 * 2, 256, 3, 1)
        self.upconv3 = self.up_block(256 * 2, 128, 3, 1)
        self.upconv2 = self.up_block(128 * 2, 64, 3, 1)
        self.upconv1 = self.up_block(64 * 2, out_channels, 3, 1)

    def down_block(self, in_channels, out_channels, kernel_size, padding):
        layer = torch.nn.Sequential(
            # First convolution
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            
            # Second convolution
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            
            # Max-pooling
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return layer

    def up_block(self, in_channels, out_channels, kernel_size, padding):
        layer = torch.nn.Sequential(
            # First convolution 
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),

            # Second convolution
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, 
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),

            # Up-convolution
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, 
                stride=2, padding=1, output_padding=1) 
        ) 
        return layer

    def crop(self, x, ref):
        if x.shape != ref.shape:
            crop_h = ref.shape[2] - x.shape[2] 
            crop_up = int(np.floor(crop_h / 2))
            crop_down = crop_h - crop_up
            crop_w = ref.shape[3] - x.shape[3] 
            crop_left = int(np.floor(crop_w / 2))
            crop_right = crop_w - crop_left
            x = F.pad(x, (crop_left, crop_right, crop_up, crop_down))
        return x

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        upconv5 = self.upconv5(conv5)
        upconv5 = self.crop(upconv5, conv4)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv4 = self.crop(upconv4, conv3)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv3 = self.crop(upconv3, conv2)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv2 = self.crop(upconv2, conv1)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        upconv1 = self.crop(upconv1, x)

        return F.log_softmax(upconv1, dim=1)


class VectorUnet(torchseg.base.BaseModel):
    """
    @brief Version of the SimpleUnet but without downsampling/upsampling.
    @details Four levels with 32, 64, 128, and 128 neurons. 
    """

    def __init__(self, in_channels=3, out_channels=35):
        """
        @param[in]  in_channels  Number of input channels, e.g. set it to
                                        three for RGB images.
        @param[in]  out_channels Number of classes.
        """
        super().__init__()
        
        # Fake way down
        self.conv1 = self.conv_block(in_channels, 64, 1, 0)
        self.conv2 = self.conv_block(64, 128, 1, 0)
        self.conv3 = self.conv_block(128, 256, 1, 0)
        self.conv4 = self.conv_block(256, 512, 1, 0)
        self.conv5 = self.conv_block(512, 1024, 1, 0)
        
        # Fake way up
        self.upconv5 = self.conv_block(1024, 512, 1, 0)
        self.upconv4 = self.conv_block(512 * 2, 256, 1, 0)
        self.upconv3 = self.conv_block(256 * 2, 128, 1, 0)
        self.upconv2 = self.conv_block(128 * 2, 64, 1, 0)
        self.upconv1 = self.conv_block(64 * 2, out_channels, 1, 0, scoring=True)

    def conv_block(self, in_channels, out_channels, kernel_size, padding,
            scoring=False):
        # Create list of operations
        modules = [
            # First convolution
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.ReLU(),
            
            # Second convolution
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1,
                            padding=padding),
            torch.nn.ReLU(),
        ]
        
        # If we are not before softmax, we use BN+ReLU as last operation
        if scoring:
            modules.append(torch.nn.BatchNorm2d(out_channels))
            modules.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding))
        #    modules.append(torch.nn.ReLU())

        layer = torch.nn.Sequential(*modules)

        return layer

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        return F.log_softmax(upconv1, dim=1)


class DeepLabV3(torchseg.base.BaseModel):

    class ASPPConv(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels, dilation):
            modules = [
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                                dilation=dilation, bias=False),
                # FIXME: commented out because it does not support bs=1
                #torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            ]
            super(DeepLabV3.ASPPConv, self).__init__(*modules)

    class ASPPPooling(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels):
            modules = [
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                # FIXME: commented out because it does not support bs=1
                #torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            ]
            super(DeepLabV3.ASPPPooling, self).__init__(*modules)

        def forward(self, x):
            size = x.shape[-2:]
            for mod in self:
                x = mod(x)
            return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


    class ASPP(torch.nn.Module):
        def __init__(self, in_channels, atrous_rates, out_channels=256):
            super(DeepLabV3.ASPP, self).__init__()
            modules = []
            modules.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                # FIXME: commented out because it does not support bs=1
                #torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()))

            rates = tuple(atrous_rates)
            for rate in rates:
                modules.append(DeepLabV3.ASPPConv(in_channels, out_channels, rate))

            modules.append(DeepLabV3.ASPPPooling(in_channels, out_channels))

            self.convs = torch.nn.ModuleList(modules)

            self.project = torch.nn.Sequential(
                torch.nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
                # FIXME: commented out because it does not support bs=1
                #torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5))

        def forward(self, x):
            res = []
            for conv in self.convs:
                res.append(conv(x))
            res = torch.cat(res, dim=1)
            return self.project(res)

    class DeepLabHead(torch.nn.Sequential):
        def __init__(self, in_channels, num_classes):
            super(DeepLabV3.DeepLabHead, self).__init__(
                DeepLabV3.ASPP(in_channels, [12, 24, 36]),
                torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                #torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, num_classes, 1)
            )

    def __init__(self, in_channels: int = 3, out_channels: int = 35,
                 pretrained: bool = False) -> None:
        """
        @param[in]  in_channels          Number of channels of the input image.
        @param[in]  out_channels         Number of output classes.
        """
        super().__init__()
        
        # Get ResNet-101 encoder
        self.backbone = torchvision.models.resnet.resnet101(
            pretrained=True, replace_stride_with_dilation=[False, True, True]) 

        # Freeze ResNet-101 encoder layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get segmentation model
        return_layers = {'layer4': 'out'}
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(self.backbone,
            return_layers=return_layers)
        self.decoder = DeepLabV3.DeepLabHead(2048, out_channels)

        # TODO: Load pretrained weights from a DeepLabV3 trained on MS-COCO (train2017)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder: get latent space vector
        input_shape = x.shape[-2:]
        latent_tensor_dict = self.backbone(x)

        # Decoder: get per-class output scores
        raw_decoder_output = self.decoder(latent_tensor_dict['out'])

        # Final touch: interpolate segmentation maps to the size of the 
        # original image
        interp_decoder_output = F.interpolate(raw_decoder_output, 
                                              size=input_shape, mode='bilinear', 
                                              align_corners=False)

        return F.log_softmax(interp_decoder_output, dim=1)


class ResNet_18_CAM_DS(torchseg.base.BaseModel):
    """
    @brief   Classification network that produces segmentation predictions as
             an intermediary step.
    @details Implementation of the architecture presented in 
             "Intrapapillary capillary loop classification in magnification 
              endoscopy: open dataset and baseline methodology" by 
              Luis C. Garcia Peraza Herrera et al., 2020.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 35,
                 pretrained: bool = False) -> None:
        """
        @param[in]  in_channels  Number of input channels, e.g. set it to
                                 three for RGB images.
        @param[in]  out_channels Number of classes.
        """
        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained

        # Get ResNet-18 encoder and modify the number of input channels
        self.resnet = torchvision.models.resnet.resnet18(pretrained=self.pretrained)
        self.resnet.conv1 = torch.nn.Conv2d(self.in_channels, 64, 
                                            kernel_size=7, stride=2, padding=3, 
                                            bias=False)

        # Create "segmentation" layers
        self.seg1 = torch.nn.Conv2d(64, self.out_channels, kernel_size=1)
        self.seg2 = torch.nn.Conv2d(64, self.out_channels, kernel_size=1)
        self.seg3 = torch.nn.Conv2d(128, self.out_channels, kernel_size=1)
        self.seg4 = torch.nn.Conv2d(256, self.out_channels, kernel_size=1)
        self.seg5 = torch.nn.Conv2d(512, self.out_channels, kernel_size=1)

        # Create GAP layers
        self.gap1 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gap2 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gap3 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gap4 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gap5 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Add the extra stuff to make ResNet-18-CAM-DS
        ilg = torchvision.models._utils.IntermediateLayerGetter
        self.conv1_tensor = ilg(self.resnet, return_layers={'bn1': 'out'})
        self.conv2_tensor = ilg(self.resnet, return_layers={'layer1': 'out'})
        self.conv3_tensor = ilg(self.resnet, return_layers={'layer2': 'out'})
        self.conv4_tensor = ilg(self.resnet, return_layers={'layer3': 'out'})
        self.conv5_tensor = ilg(self.resnet, return_layers={'layer4': 'out'})
        
        # Create layer to convert scores into fake probabilities
        self.log_sigmoid = torch.nn.LogSigmoid()
        
    def forward(self, x: torch.Tensor):
        """
        @returns a dictionary of log-probability predictions from the side
                 branches and the global branch. The key 'logprob6_output'
                 represents the global prediction. The tensors should be 
                 one-dimensional, with each element indicating the probability
                 of that class being in the image.
        """

        # Get tensors at every resolution level
        conv1_output = self.conv1_tensor(x)['out']
        conv2_output = self.conv2_tensor(x)['out']
        conv3_output = self.conv3_tensor(x)['out']
        conv4_output = self.conv4_tensor(x)['out']
        conv5_output = self.conv5_tensor(x)['out']

        # Produce a feature map per class at every resolution level
        seg1_output = self.seg1(conv1_output)
        seg2_output = self.seg2(conv2_output)
        seg3_output = self.seg3(conv3_output)
        seg4_output = self.seg4(conv4_output)
        seg5_output = self.seg5(conv5_output)

        # Compute class scores (with GAP) at every resolution level 
        # (and globally too)
        score1_output = self.gap1(seg1_output)
        score2_output = self.gap2(seg2_output)
        score3_output = self.gap3(seg3_output)
        score4_output = self.gap4(seg4_output)
        score5_output = self.gap5(seg5_output)
        score6_output = score1_output + score2_output + score3_output \
                        + score4_output + score5_output

        # Compute sigmoid function to estimate the side "probabilities"
        logprob1_output = self.log_sigmoid(score1_output)
        logprob2_output = self.log_sigmoid(score2_output)
        logprob3_output = self.log_sigmoid(score3_output)
        logprob4_output = self.log_sigmoid(score4_output)
        logprob5_output = self.log_sigmoid(score5_output)

        # Compute the global "probabilities"
        logprob6_output = self.log_sigmoid(score6_output)

        # Stack the class-presence log-probability predictions at every 
        # resolution level
        class_presence = torch.stack([logprob1_output, logprob2_output, 
                                      logprob3_output, logprob4_output, 
                                      logprob5_output, logprob6_output])
        
        return class_presence, seg1_output, seg2_output, seg3_output, seg4_output, seg5_output


if __name__ == '__main__':
    raise RuntimeError('The model.py module is not is not a script.') 
