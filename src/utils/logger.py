import logging
import logging.config
import yaml
from pathlib import Path

def setup_logger():
    config_path = Path("config/logging.yaml")

    with open(config_path,"r") as file:
        logging_config = yaml.safe_load(file)
    
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

    return logger