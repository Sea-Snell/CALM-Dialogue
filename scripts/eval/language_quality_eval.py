import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from load_objs import load_item
from accelerate import Accelerator
from models.base import Policy
from utils.logs_utils import DistributeCombineLogs
from utils.misc import add_system_configs
from torch.utils.data import DataLoader
from utils.misc import convert_path
import json
from tqdm.auto import tqdm

def language_eval(cfg):
    print('using config:', cfg)
    cfg['output_dir'] = convert_path(cfg['output_dir'])
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])
    
    ad_dataset_eval = load_item(cfg['dataset'])
    eval_data_loader_kwargs = {'num_workers': cfg['dataloader_workers'], 
                               'batch_size': cfg['bsize'], 
                               'collate_fn': ad_dataset_eval.collate}
    eval_data_loader = DataLoader(ad_dataset_eval, **eval_data_loader_kwargs)
    evaluator = load_item(cfg['evaluator'], system_cfg['device'])

    model = load_item(cfg['model'], system_cfg['device'])
    model.eval()
    if not isinstance(model, Policy):
        model, eval_data_loader = accelerator.prepare(model, eval_data_loader)
    else:
        eval_data_loader = accelerator.prepare(eval_data_loader)

    eval_logs = DistributeCombineLogs(accelerator, use_wandb=False)
    eval_logs.reset_logs()

    with torch.no_grad():
        for i, eval_scenes in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader)):
            if cfg['eval_batches'] != -1 and i >= cfg['eval_batches']:
                break
            logs = {}
            if cfg['call_get_loss']:
                _, logs, _ = accelerator.unwrap_model(model).get_loss(eval_scenes, **cfg['loss'])
            logs['evaluation'] = evaluator.evaluate(accelerator.unwrap_model(model), eval_scenes)
            eval_logs.accum_logs(logs)
    eval_total_logs = evaluator.postproc(eval_logs.log())
    print(eval_total_logs)
    with open(cfg['output_dir'], 'w') as f:
        json.dump(eval_total_logs, f)

@hydra.main(config_path="../../config", config_name="language_eval")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    language_eval(cfg)

if __name__ == "__main__":
    main()