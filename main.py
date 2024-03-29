from omegaconf import DictConfig, ListConfig, OmegaConf
import argparse
from yolo import YoloS
from dataset import build_dataset
from trainer import YoloTrainer
import logging


def select_log_level(cfg):
    """ Selects the log level based on the input string
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(level=levels[cfg.experiment.log_level])


def get_config():
	"""	Creates a config object from the yaml file and the cli arguments
	"""
	cli_conf = OmegaConf.from_cli()

	yaml_conf = OmegaConf.load(cli_conf.config)
	conf = OmegaConf.merge(yaml_conf, cli_conf)
	return conf


if __name__=="__main__":
    
    cfg = get_config()
    
    # log level
    select_log_level(cfg)
    
    # build the model, data loader and trainer
    model = YoloS()
    data_loaders = build_dataset(cfg)
    trainer = YoloTrainer(cfg, model, data_loaders)
    
    # train the model
    trainer.train()