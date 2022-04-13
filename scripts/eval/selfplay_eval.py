import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item
from selfplay import selfplay
from utils.misc import convert_path, mp_device
import torch.multiprocessing as mp
import json
import torch
from tqdm.auto import tqdm

class Worker:
    def __init__(self, q, cfg) -> None:
        device = mp_device().device
        print('num processes:', mp_device().num)
        print('using device:', device)
        self.customer_bot = load_item(cfg['customer_bot'], device)
        self.customer_bot.eval()
        self.agent_bot = load_item(cfg['agent_bot'], device)
        self.agent_bot.eval()
        self.max_turns = cfg['selfplay']['max_turns']
        self.verbose = cfg['selfplay']['verbose']
        self.q = q
    
    def process(self, scene):
        reward, curr_event = selfplay(self.customer_bot, self.agent_bot, scene, self.max_turns, self.verbose)
        if self.verbose:
            print()
            print()
        result = {'reward': reward, 
                  'scene': {'meta': scene.data, 'scene_id': scene.scene_id}, 
                  'dialogue': list(map(lambda x: (str(x.agent), str(x.event)), curr_event.get_all_events()))}
        self.q.put(result)
        return 0

def listener(output_file, q):
    '''listens for messages on the q, writes to file. '''
    with open(output_file, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(json.dumps(m) + '\n')
            f.flush()

def init(q, cfg):
    global worker
    worker = Worker(q, cfg)

def process(item):
    global worker
    return worker.process(item)

def selplay_eval(cfg):
    print('using config:', cfg)
    selfplay_config = cfg['selfplay']
    selfplay_config['outputs_file'] = convert_path(selfplay_config['outputs_file'])
    selfplay_config['load_outputs_file'] = convert_path(selfplay_config['load_outputs_file'])
    global worker
    worker = None

    if torch.cuda.is_available():
        mp.set_start_method('spawn')
        mp.set_sharing_strategy('file_system')

    if selfplay_config['load_outputs_file'] is not None:
        saved_items = [json.loads(item) for item in open(selfplay_config['load_outputs_file'], 'r')]
        saved_ids = set([item['scene']['scene_id'] for item in saved_items])
    else:
        saved_items = []
        saved_ids = set([])

    print('setting up...')
    q = mp.Manager().Queue()
    p = mp.Process(target=listener, args=(selfplay_config['outputs_file'], q,))
    p.start()

    for item in saved_items:
        q.put(item)
    print('loaded previously completed conversation ids:', saved_ids)

    data = load_item(cfg['dataset'])

    scenes = []
    for scene in data:
        if scene.scene_id not in saved_ids:
            scenes.append(scene)

    with mp.Pool(mp_device().num, initializer=init, initargs=(q, cfg)) as pool:
        data = list(tqdm(pool.imap(process, scenes), total=len(scenes)))
        q.put('kill')

    p.join()
    print('done.', data)

@hydra.main(config_path="../../config", config_name="selplay_eval")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    selplay_eval(cfg)

if __name__ == "__main__":
    main()