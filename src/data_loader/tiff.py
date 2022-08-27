# Copyright © 2020 University of Eastern Finland
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Methods to read and write spectral tiff image cubes.
#
# Part of our UEF Computational Spectral Imaging group has been advocating
# Tiff file format for spectral image storage. The main benefit of this
# multipage image format is that it is widely supported. This means that
# the file managers of the major operating systems can show a thumbnail
# preview of the first page in the Tiff image file. The format also allows
# the use of custom tags in the file descriptor metadata. This allows the
# group to embed the center wavelength list of the spectral image bands
# among other metadata.
#
# The spectral image tiffs the group produces have following layout:
#   1st page: RGB render of the spectral image cube under D65 illuminant,
#   2nd page - nth page: the band images of the spectral image cube.
# Tiff tag 65000 contains the wavelength band center wavelengths as a list of
# 32-bit floating point values.
# Tiff tag 65111 contains free-form metadata of the imaging conditions,
# sample, etc., as an ASCII string.
#
# This file also contains two functions, read_mtiff and write_mtiff, for
# storage and retrieval of mask bitmaps. Each page of the tiff file contains
# a bitmap image of an annotation mask, and each page is associated with
# tiff tag 65001 containing the mask label string, again as an ASCII string.
#
# The read_stiff and write_stiff methods provided in this file are based on
# the tifffile-package. The older read_tiff and write_tiff were utilizing
# the same package, but it was embedded in scikit-image. The scikit-image
# package in turn has heavy dependencies, which make it unsuitable in
# certain environments. The tifffile package on the other hand is much
# lighter and more suitable in small environments.
#
# Tifffile package changed how Tiff tags are exposed in version 2020.2.16.
# This version has not made its way into the main repositories in Anaconda
# distribution, but it is available in the conda-forge repository.
# Installation instructions for Anaconda:
#   conda install -c conda-forge tifffile=2020.2.16
#
# Alternatively, recreate the virtual environment suitable for using these
# functions with pipenv:
#   pipenv install
#
# 2020-06-25: Add read_mtiff, write_mtiff for reading and writing mask bitmaps.
# 2020-06-02: Add rgb_only-parameter to allow loading of the RGB-render only.
# 2020-05-27: Add silent-parameter to silence warnings about duplicate
#             metadata and wavelength lists.
# 2020-03-26: Initial version of the read_stiff and write_stiff functions
#             based on the previous read_tiff and write_tiff functions by
#             Pauli Fält and myself.
#             -- Joni Hyttinen
#

from typing import Optional, Any
import warnings

from tifffile import TiffFile, TiffWriter
import numpy as np


def read_stiff(filename: str, silent=False, rgb_only=False):
    """

    :param filename:    filename of the spectral tiff to read.
    :return:            Tuple[spim, wavelengths, rgb, metadata], where
                        spim: spectral image cube of form [height, width, bands],
                        wavelengths: the center wavelengths of the bands,
                        rgb: a color render of the spectral image [height, width, channels] or None
                        metadata: a free-form metadata string stored in the image, or an empty string
    """
    TIFFTAG_WAVELENGTHS = 65000
    TIFFTAG_METADATA = 65111
    spim = None
    wavelengths = None
    rgb = None
    metadata = None

    first_band_page = 0
    with TiffFile(filename) as tiff:
        # The RGB image is optional, the first band image maybe on the first page:
        first_band_page = 0
        if tiff.pages[first_band_page].ndim == 3:
            rgb = tiff.pages[0].asarray()
            # Ok, the first band image is on the second page
            first_band_page = first_band_page + 1

        multiple_wavelength_lists = False
        multiple_metadata_fields = False
        for band_page in range(first_band_page, len(tiff.pages)):
            # The wavelength list is supposed to be on the first band image.
            # The older write_tiff writes it on all pages, though, so make
            # a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_WAVELENGTHS)
            tag_value = tag.value if tag else tuple()
            if tag_value:
                if wavelengths is None:
                    wavelengths = tag_value
                elif wavelengths == tag_value:
                    multiple_wavelength_lists = True
                elif wavelengths != tag_value:
                    # Well, the image is just broken then?
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing wavelength lists!')

            # The metadata string, like the wavelength list, is supposed to be
            # on the first band image. The older write_tiff wrote it on all
            # pages, too. Make a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_METADATA)
            tag_value = tag.value if tag else ''
            if tag_value:
                if metadata is None:
                    metadata = tag_value
                elif metadata == tag_value:
                    multiple_metadata_fields = True
                elif metadata != tag_value:
                    # Well, for some reason there are multiple metadata fields
                    # with varying content. This version of the function does
                    # not care for such fancyness.
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing metadata fields!')

        # The metadata is stored in an ASCII string. It may contain back-slashed
        # hex sequences (unicode codepoints presented as ASCII text). Convert
        # ASCII string back to bytes and decode as unicode sequence.
        if metadata:
            metadata = metadata.encode('ascii').decode('unicode-escape')
        else:
            metadata = ''

        # Some of the early images may have errorneus metadata string.
        # Attempt to fix it:
        if metadata[0] == "'" and metadata[-1] == "'":
            while metadata[0] == "'":
                metadata = metadata[1:]
            while metadata[-1] == "'":
                metadata = metadata[:-1]
            if '\\n' in metadata:
                metadata = metadata.replace('\\n', '\n')

        # Generate a fake wavelength list, if the spectral tiff has managed to
        # lose its own wavelength list.
        if not wavelengths:
            wavelengths = range(0, len(tiff.pages) - 1 if rgb is not None else len(tiff.pages))

        if multiple_wavelength_lists and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated wavelength lists!')
        if multiple_metadata_fields and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated metadata fields!')

        if not rgb_only:
            spim = tiff.asarray(key=range(first_band_page, len(tiff.pages)))
            spim = np.transpose(spim, (1, 2, 0))
        else:
            spim = None

        # Make sure the wavelengths are in an ascending order:
        if wavelengths[0] > wavelengths[-1]:
            spim = spim[:, :, ::-1] if spim is not None else None
            wavelengths = wavelengths[::-1]

    # Convert uint16 cube back to float32 cube
    if spim is not None and spim.dtype == 'uint16':
        spim = spim.astype('float32') / (2**16 - 1)

    return spim, np.array(wavelengths), rgb, metadata


def write_stiff(filename: str, spim, wls, rgb: Optional[Any], metadata: str = ''):
    """
    Write a spectral image cube into a Spectral Tiff. A spectral tiff contains
    two custom tags to describe the data cube:
        - wavelength list is stored in tag 65000 as a list of float32s, and
        - a metadata string is stored in tag 65111 as a UTF-8 encoded byte string.

    :param filename:    the filename of the spectral tiff to save the data cube in
    :param spim:        the spectral image data cube, expected dimensions [height, width, bands]
    :param wls:         the wavelength list, length of the list must match number of bands
    :param rgb:         color image render of the spectral image cube. This is
                        saved as the first page of the spectral tiff. Many file
                        managers choose to show the first page of the tiff image
                        as a preview/thumbnail. This parameter is optional.
    :param metadata:    a free-form metadata string to be saved in the spectral tiff.
    """
    if wls.dtype != 'float32':
        warnings.warn(f'Wavelength list dtype {wls.dtype} will be saved as float32. Precision may be lost.')
        wls = wls.astype('float32')
    wavelengths = list(wls)
    metadata_bytes = str(metadata).encode('ascii', 'backslashreplace')
    stiff_tags = [
        (65000, 'f', len(wavelengths), wavelengths, True),
        (65111, 's', len(metadata_bytes), metadata_bytes, True)
    ]

    if len(wls) != spim.shape[2]:
        raise ValueError(f'Wavelength list length {len(wls)} does not match number of bands {spim.shape[2]}')

    # RGB image must have three channels and dtype uint8
    if rgb is not None and rgb.ndim != 3:
        raise TypeError(f'RGB preview image must have three channels! (ndim = {rgb.ndim} != 3)')

    if rgb is not None and rgb.dtype != 'uint8':
        warnings.warn(f'RGB preview image is not a uint8 array (dtype: {rgb.dtype}).')
        if rgb.dtype == 'float':
            rgb = (rgb * (2**8-1)).astype('uint8')
        else:
            raise RuntimeError(f'How should {rgb.dtype} be handled here?')

    with TiffWriter(filename) as tiff:
        if rgb is not None:
            tiff.save(rgb)

        # Save the first page with tags
        spim_page = spim[:, :, 0]
        tiff.save(spim_page, extratags=stiff_tags)

        # continue saving pages
        for i in range(1, spim.shape[2]):
            spim_page = spim[:, :, i]
            tiff.save(spim_page)


def read_mtiff(filename):
    """
    Read a mask bitmap tiff.

    Mask bitmap tiff contains multiple pages of bitmap masks. The mask label
    is stored in tag 65001 in each page. The mask label is stored as an ASCII
    string that may contain unicode codepoints encoded as ASCII character
    sequences (see unicode-escape encoding in Python docs).

    :param filename:    filename of the mask tiff to read.
    :return:            Dict[label: str, mask: ndarray], where
                        label: the mask label
                        mask: the boolean bitmap associated with the label.
    """
    TIFFTAG_MASK_LABEL = 65001
    masks = dict()
    with TiffFile(filename) as tiff:
        for p in range(0, len(tiff.pages)):
            label_tag = tiff.pages[p].tags.get(TIFFTAG_MASK_LABEL)
            label = label_tag.value.encode('ascii').decode('unicode-escape')
            mask = tiff.asarray(key=p)
            masks[label] = mask > 0
    return masks


def write_mtiff(filename, masks):
    """
    Write a mask bitmap tiff.

    Mask bitmap tiff contains multiple pages of bitmap masks. The mask label
    is stored in tag 65001 in each page. The mask label is stored as an ASCII
    string that may contain unicode codepoints encoded as ASCII character
    sequences (see unicode-escape encoding in Python docs).

    :param filename:    filename of the mask tiff to write to.
    :param masks:       Dict[label: str, mask: ndarray], where
                        label: the mask label
                        mask: the boolean bitmap associated with the label.
    """
    with TiffWriter(filename) as tiff:
        for label in masks:
            label_bytes = str(label).encode('ascii', 'backslashreplace')
            tiff.save(masks[label] > 0, #.astype('uint8') * 255,
                      photometric='MINISBLACK',
                      contiguous=False,
                      extratags=[(65001, 's', len(label_bytes), label_bytes, True)])
