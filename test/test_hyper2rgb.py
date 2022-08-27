"""
@brief  Unit tests for the reconstruction of RGB images from hyperspectral data.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   9 Feb 2022.
"""

import unittest
import numpy as np
import colour
import scipy

# My imports
import torchseg.data_loader

class TestReconstructionMethods(unittest.TestCase):

    def test_reconstruction_of_nuance_d65(self):
        # Get function for the D65 illuminant
        il = colour.SDS_ILLUMINANTS['D65']
        f_illum = scipy.interpolate.PchipInterpolator(il.wavelengths, il.values, extrapolate=True)

        # Generate a hyperspectral image of 51 bands containing the D65 illuminant  
        min_wl = 450
        max_wl = 950
        nbands = max_wl - min_wl
        bands = np.linspace(min_wl, max_wl, nbands)
        h = 1
        w = 1
        im_hyper = np.empty((h, w, nbands))
        im_hyper[:, :, :] = f_illum(bands)

        # Convert illuminant image to RGB
        image_loader = torchseg.data_loader.OdsiDbDataLoader.LoadImage
        im_rgb = image_loader.hyper2rgb(im_hyper, bands)

        print(im_rgb)
        print(im_rgb.shape)
        

if __name__ == '__main__':
    unittest.main()
