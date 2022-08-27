"""
@brief  Collection of functions for general purpose use. 
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021. 
"""

import os
import re
import pathlib
import json
import collections
import torch
import pandas as pd


## ----- Functions ----- ##


def read_json(fname):
    """@brief Reads a JSON file and returns a dictionary."""
    fname = pathlib.Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
    """@brief Dumps a dictionary into a JSON file."""
    fname = pathlib.Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def setup_gpu_devices(n_gpu_use):
    """
    @brief If available, setup GPU devices. Otherwise, CPU will be used.
    @param[in]  n_gpu_use  Number of GPUs you want to use. 
    @returns a tuple (torch.device, list) where the list has the IDs of the GPUs
             that will be used.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("""Warning: There\'s no GPU available on this machine,
                 training will be performed on CPU.""")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"""Warning: The number of GPU\'s configured to use is {n_gpu_use}, 
                  but only {n_gpu} are available on this machine.""")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

 
def natsort(l): 
    """
    @brief Natural sort of a list, i.e. using alphabetic order.
    @param[in]  l  List to sort. 
    @returns A new list sorted taking into account numbers and not just their 
             ASCII codes.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', 
        str(key)) ]
    return sorted(l, key = alphanum_key)


def listdir(path, hidden=False):
    """
    @brief Lists a directory removing the extension of the files and the 
           hidden files.
    @param[in]  path    String containing the path (relative or absolute) to 
                        the folder that you want to list.
    @param[in]  hidden  Set to True if you want to list the hidden files too.
                        By default, it is set to False.
    @returns A list of visible files and folders inside the given path.
    """
    files = []
    if hidden:
        files = natsort(os.listdir(path)) 
    else:
        files = natsort([f for f in os.listdir(path) if not f.startswith('.')])
    return files

## ----- Classes ----- ##


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, 
            columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


