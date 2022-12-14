"""
@brief  Module with the definition of the base data loader. 
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com). 
@date   1 Jun 2021.
"""

import numpy as np
import torch


class BaseDataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size, shuffle, validation_split, 
            num_workers, collate_fn=torch.utils.data.dataloader.default_collate):
        """
        @brief Base class for all data loaders.
        @param[in]  dataset           TODO.
        @param[in]  batch_size        Training and validation batch size. 
        @param[in]  shuffle           Boolean flag for shuffling data that is 
                                      currently not used.
        @param[in]  validation_split  TODO. 
        @param[in]  num_workers       Number of PyTorch workers. That is, how 
                                      many sub-processes to use for data loading.
        @param[in]  collate_fn        Function that merges a list of samples to 
                                      form a mini-batch of tensor[s].
        """
        # Sanity check: the dataset must have a training flag, cause we use it
        #               here in the BaseDataLoader
        #assert(hasattr(dataset, 'training'))

        # Store parameters into attributes
        self.shuffle = shuffle
        self.validation_split = validation_split

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = \
            self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert(split > 0 and split < self.n_samples)
            #"validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        # Turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return torch.utils.data.DataLoader(sampler=self.valid_sampler, 
                **self.init_kwargs)

    #@property 
    #def training(self):
    #    return self.dataset.training

    #@training.setter
    #def training(self, training):
    #    self.dataset.training = training


