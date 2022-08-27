#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import unittest

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='torchseg',
    version='0.1.0',
    description='Python module to segment hyperspectral dental images.',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT License',
    url='https://github.com/luiscarlosgph/dentalseg',
    packages=[
        'torchseg',
        'torchseg.base',
        'torchseg.config',
        'torchseg.data_loader',
        'torchseg.logger',
        'torchseg.model',
        'torchseg.machine',
        'torchseg.utils',
        'torchseg.visualization',
    ],
    package_dir={
        'torchseg':               'src',
        'torchseg.base':          'src/base',
        'torchseg.config':        'src/config',
        'torchseg.data_loader':   'src/data_loader',
        'torchseg.logger':        'src/logger',
        'torchseg.model':         'src/model',
        'torchseg.machine':       'src/machine',
        'torchseg.utils':         'src/utils',
        'torchseg.visualization': 'src/visualization',
    },
    install_requires = [
        'numpy', 
        'opencv-python',
        'torch',
        'torchvision',
        'tensorboard',
        'tqdm',
        'tifffile',
        'monai',
        'pandas',
        'colour-science',
        'matplotlib',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    test_suite = 'test',
)
