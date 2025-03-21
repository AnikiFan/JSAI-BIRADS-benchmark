from utils.multiMessageFilter import MultiMessageFilter  # ! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
from config.config import init_config, Config
from utils.Trainer import Trainer
from hydra import main
from omegaconf import OmegaConf

@main(version_base=None, config_name="config",config_path="config")
def main(cfg:Config):
    trainer = Trainer(cfg)
    return trainer.train()

if __name__ == '__main__':
    MultiMessageFilter().setup()
    init_config()
    main()
