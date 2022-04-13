import hydra
from omegaconf import DictConfig, OmegaConf
from supervised_train_loop import train

@hydra.main(config_path="../../config", config_name="train_real_table_agent")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()