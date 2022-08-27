"""
@brief  Module to setup and maintain the logging abilities of the training and 
        validation scripts.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021.
"""

import logging
import logging.config
import pathlib
import torchseg.utils

class LoggerSetup():
    def __init__(self, save_dir, log_config, default_level=logging.INFO):
        """
        @brief Setup logging configuration.
        """
        log_config = pathlib.Path(log_config)
        if log_config.is_file():
            config = torchseg.utils.read_json(log_config)
            
            # Modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])
            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration file is not found \
                in {}.".format(log_config))
            logging.basicConfig(level=default_level)
