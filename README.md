Introduction
------------
This repository contains code and data related to the paper ["Hyperspectral image segmentation: A preliminary study on the Oral and Dental Spectral Image Database (ODSI-DB)", Luis Carlos Garcia Peraza Herrera et al., MICCAI 2022 16th AE-CAI, 9th CARE and 5th OR 2.0 Workshop](). Also available in [Arxiv]().


Download links
--------------

* Digital Poster file (PDF): [here](doc/Paper#22_GarciaPerazaHerrera_Poster.pdf).
* Short slide deck for short presentation: PPT [here]() and PDF [here]().
* Final camera-ready PDF: [here]().


Citation
--------
Please if you use this code in your work, cite:
```
@article{Garcia-Peraza-Herrera2022,  
  author={Garcia-Peraza-Herrera, Luis C. and Horgan, Conor and Ourselin, Sebastien and Ebner, Michael and Vercauteren, Tom},  
  journal={},   
  title={Hyperspectral image segmentation: A preliminary study on the Oral and Dental Spectral Image Database (ODSI-DB)},   
  year={2022},  
  volume={},  
  number={},  
  pages={},  
  doi={}
}
```
<!-- Hyperspectral segmentation of dental images from the [ODSI-DB dataset](https://sites.uef.fi/spectral/odsi-db). -->

Dependencies
------------

* PyTorch Stable (1.8.1): 

```bash
$ python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html --user
```

* MONAI (0.5.3):

```bash
$ python3 -m pip install monai --user
```


Download the cross-validation folds for training/testing
--------------------------------------------------------
The cross-validation folds used in the study can be downloaded [here](https://www.synapse.org/#!Synapse:syn26141810/files).


Install from source
-------------------

```bash
$ python3 setup.py install --user
```

Generate training/testing splits
--------------------------------
Script to create cross-validation folds. Each fold will contain a different train/test partition of dataset images. 

Assuming that you have the ODSI-DB dataset stored in ```~/odsi_db```, you can generate the training/testing folds running:

```bash
$ python3 -m torchseg.generate_odsi_db_split --input ~/odsi_db/dataset --output ~/odsi_db/folds --folds 5 --train 0.9
```
Arguments:
* ```--folds```: number of train/test partitions to generate.
* ```--train```: ratio of images that will be added to the training split of each fold.

In the generated folds, all the classes will be present in the images of the training split. However, not all the classes will be present in the testing split (simply because the split is done by images and there are classes whose pixels can be found only in a single image of the dataset).

Note that the further partition of the training split into train/val is done on the fly during training. The aim of the ```torchseg.generate_odsi_db_split``` script described in this section is simply to put aside some images for testing.

Run training and validation
---------------------------

Trained models are saved in the directory specified by the key ```trainer/save_dir``` in the config JSON file.
In the default configuration templates [provided](https://github.com/luiscarlosgph/dentalseg/tree/main/config_templates) this 
is defined as: ```"saved/"```. Therefore the models saved during training will be located inside ```./saved/models```.

Classification example:

```bash
# MNIST 
$ python3 -m torchseg.train -c config_templates/classification/mnist.json -l logger_config.json
```


Segmentation example:

```bash
# ODSI-DB
$ python3 -m torchseg.train -c config_templates/segmentation/odsidb.json -l logger_config.json
```

Run testing
-----------

Classification example:

```bash
# MNIST 
$ python3 -m torchseg.test -c config_templates/classification/mnist.json -l logger_config.json -r model_best.pth
```


Segmentation example:

```bash
$ python3 -m torchseg.test -c config.json -l logger_config.json -r model_best.pth -o results.json
```

<!--
Run inference on a single image
-------------------------------
TODO
-->


Print ODSI-DB stats
-------------------

Run this script to print some of the ODSI-DB statistics: number of images recorded with Nuance EX (51 bands), number of images recorded with Specim IQ (204 bands), number of annotated images, number of annotated pixels per class, number of images in which each class is represented.

```bash
$ python3 -m torchseg.odsi_db_stats -i ~/odsi_db/dataset
```


How to split the images of the dataset by camera model
------------------------------------------------------
```
$ python3 -m torchseg.split_images_per_camera_model -i odsi_db/dataset -o odsi_db/dataset_by_camera_type
```


Generate reconstructed RGB images from the hyperspectral ones provided in the dataset
-------------------------------------------------------------------------------------
* If you want to get a folder of RGB reconstructions versus RGB provided in the dataset:
```bash
$ python3 -m torchseg.validate_dataset -i datasets/odsi_db/dataset -o datasets/odsi_db/provided_rgb_vs_reconstructed_rgb
```
* If you want to get a folder containing the RGB reconstructions for each camera model in the dataset:
```bash
$ python3 -m torchseg.generate_rgb_recon -i datasets/odsi_db/dataset -o experiments/odsi_db/recon_rgb
```Red average: 0.403129887605
Green average: 0.288007343691
Blue average: 0.299380141092

Generate t-SNE of RGB and hyperspectral pixels in the dataset
-------------------------------------------------------------
```bash
$ python3 -m torchseg.generate_tsne -i <dataset input dir path> -o <output dir path> -n <number of pixels> -r <use reconstructed RGB> -v <only visible range> -e <interpolation mode>
```
For example:
```bash
$ python3 -m torchseg.generate_tsne -i odsi_db/dataset_by_camera_type/specim_iq -o odsi_db/recon_rgb/tsne_specim_iq -n 10000000 -r 1 -v 0 -e linear
```


Exemplary configuration file
----------------------------
The idea of the JSON config file (same for command line parameters) is to provide a way to customise the experiment without having to parse the config by hand.

Most sections in the JSON represent classes and the contents of *args* are the parameters that are subsequently passed to the constructor of such classes. That is, **type** should be a class name and **args** the parameters for the constructor of the class specified in **type**.

The classes used can be from PyTorch, MONAI or custom-made.



```
{
    # Name of the experiment
    "name": "Mnist_LeNet",
    "n_gpu": 1,

    # Name of the model class and parameters to be passed to the constructor 
    # (any model defined in src/model/model.py)
    "model": {
        "type": "MnistModel",
        "args": {}
    },  
    
    # Name of the data loader class and parameters for the constructor 
    # (any data_loader defined in src/data_loader/data_loader.py)
    "data_loader": {
        "type": "MnistDataLoader",
        "args":{
            "data_dir": "data/",
            "batch_size": 128,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 2
        }   
    },  
    
    # Name of the optimizer class and parameters for the constructor 
    # (any optimizer in torch.optim, defined in train.py)
    "optimizer": {
        "type": "SGD",
        "args":{
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.99 
        }   
    },  
    
    # Name of the loss function (any function defined in src/model/loss.py)
    "loss": "nll_loss",

    # List of names of metric functions (any function defined in src/model/metric.py)
    "metrics": [
        "accuracy", "top_k_acc"
    ],  
    
    # Name of the learning rate scheduler class (any torch.optim.lr_scheduler, 
    # defined in train.py)
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50, 
            "gamma": 0.1 
        }   
    },
    
    # Name of the learning machine (any class defined in src/machine/machine.py)
    "machine": {
        "type": "GenericMachine", 
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10, 
        "tensorboard": true
    },  
    
    # List of datasets for the testing phase, 'type' is just the name of the data 
    # loader class defined or imported in src/data_loader/data_loader.py)
    "testing": {
        "datasets": [
            {   
                "type": "MnistDataLoader", 
                "args":{
                    "data_dir": "data/",
                    "batch_size": 1,
                    "shuffle": true,
                    "validation_split": 1.0,
                    "num_workers": 1
                }   
            }   
        ]   
    }   
}
```

Tensorboard
-----------
```bash
$ tensorboard --logdir saved/log
```
The web server will open at [http://localhost:6006](http://localhost:6006).


ODSI-DB normalisation stats
---------------------------

* [RGB mean](https://github.com/luiscarlosgph/dentalseg/blob/main/data/rgb_mean.txt)
* [RGB std](https://github.com/luiscarlosgph/dentalseg/blob/main/data/rgb_std.txt)
* [Nuance EX mean](https://github.com/luiscarlosgph/dentalseg/blob/main/data/nuance_ex_mean.txt)
* [Nuance EX std](https://github.com/luiscarlosgph/dentalseg/blob/main/data/nuance_ex_std.txt)
* [Specim IQ mean](https://github.com/luiscarlosgph/dentalseg/blob/main/data/specim_iq_mean.txt)
* [Specim IQ std](https://github.com/luiscarlosgph/dentalseg/blob/main/data/specim_iq_std.txt)
 
These values are normalised to the range [0, 1]. They RGB values have been estimated from the RGB reconstructions obtained 
with the algorithm in this repository, not from the RGB reconstructions that come in ODSI-DB. 


Author
------
Copyright (C) 2022 Luis Carlos Garcia Peraza Herrera. All rights reserved.


License
-------
This project is distributed under an [MIT license](https://github.com/luiscarlosgph/dentalseg/blob/main/LICENSE).
