"""
@brief  Base class for deep learning models.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021.
"""
import torch
import numpy as np
import abc


class BaseModel(torch.nn.Module):
    """
    @brief Base class for all models.
    """

    def from_file():
        # TODO
        #model = TheModelClass(*args, **kwargs)
        #model.load_state_dict(torch.load(PATH))
        #model.eval()
        pass

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        @brief Forward pass implementation.
        @returns The model output.
        """
        raise NotImplementedError

    def __str__(self):
        """
        @brief Print model and number of trainable parameters.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
