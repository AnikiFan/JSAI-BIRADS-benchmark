from utils.multiMessageFilter import MultiMessageFilter  # ! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
from config.config import init_config, Config
from Trainer import Trainer
from hydra import main
from hydra.core.hydra_config import OmegaConf

@main(version_base=None, config_name="config")
def main(cfg:Config):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    MultiMessageFilter().setup()
    init_config()
    main()
