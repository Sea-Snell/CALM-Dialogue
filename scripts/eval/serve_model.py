from tracemalloc import start
import hydra
from omegaconf import DictConfig, OmegaConf
from ad.airdialogue import Event, Scene
from load_objs import load_item
import pickle as pkl
import redis

from flask import Flask
from flask import request
from flask_cors import CORS
import time

import multiprocessing as mp

from utils.misc import stack_dicts
import torch
import json

app = Flask(__name__)
CORS(app)
Q = None
r = None
config = None

def queue_model(device):
    global Q
    global r
    global config
    bot = load_item(config['bot'], device)
    bot.eval()
    print('loaded bot.')
    while True:
        try:
            request_id, ev, scene = Q.get()
            with torch.no_grad():
                result = bot.respond(ev, scene)
            r.set(f'result_{request_id}', pkl.dumps(result))
        except EOFError:
            return
        except Exception as e:
            raise Exception

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    return 'hello world'

@app.route('/respond', methods=['POST'])
def respond():
    global Q
    global r
    global config
    items = request.json
    scene = Scene.from_json(items['scene'])
    scene.data[config['data']['cond_reward_key']] = 1.0
    curr_ev = None
    for speaker, msg in items['history']:
        new_ev = Event.from_sentence(speaker, msg, scene)
        if curr_ev is not None:
            new_ev = curr_ev.append(new_ev)
        curr_ev = new_ev
    request_id = int(r.incr('request_id_counter'))
    Q.put((request_id, curr_ev, scene,))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(config['serve']['client_delay'])
    result = pkl.loads(r.get(f'result_{request_id}'))
    return json.dumps([str(result.agent), str(result.event)])

def start_flask():
    global serve_cfg
    serve_cfg = config['serve']
    app.run(host=serve_cfg['flask_host'], port=serve_cfg['flask_port'], 
            threaded=serve_cfg['flask_threaded'], processes=serve_cfg['flask_processes'])

def serve(cfg):
    global Q
    global r
    global config
    config = cfg
    print(config)
    serve_cfg = config['serve']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    Q = mp.Manager().Queue()
    r = redis.Redis(host=serve_cfg['redis_host'], port=serve_cfg['redis_port'], db=serve_cfg['redis_db'])

    p = mp.Process(target=start_flask)
    p.start()

    queue_model(device)

@hydra.main(config_path="../../config", config_name="serve_model")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    serve(cfg)

if __name__ == "__main__":
    mp.set_start_method('fork')
    main()

