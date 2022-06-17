import json
import logging
import logging.config

import PyTorch_classifier.controller.trainer

# load logging config
with open('log_config.json', 'r') as f:
    log_conf = json.load(f)
logging.config.dictConfig(log_conf)

if __name__ == '__main__':
    PyTorch_classifier.controller.trainer.train()
