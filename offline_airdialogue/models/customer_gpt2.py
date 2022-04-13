import torch
from models.base import GPT2LMBase, Evaluator
from ad.airdialogue import Event, Scene
from typing import Union, List, Optional, Dict, Any
from utils.sampling_utils import *
from utils.torch_utils import get_transformer_logs

class CustomerGPT2(GPT2LMBase):
    def __init__(self, 
                 gpt2_type: str = "gpt2", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None):
        super(CustomerGPT2, self).__init__(gpt2_type=gpt2_type, 
                                           device=device, 
                                           max_length=max_length)
    
    def format(self, dialogue_events: List[Event], scene: Scene):
        return ("<s> " + scene.customer_scenario.get_str_state() + " </s> " + self.format_events(dialogue_events)).strip()
    
    def get_loss(self, scenes: List[Scene]):
        tokens, attn_mask = self._tokenize_batch([scene.events for scene in scenes], scenes)
        model_outputs = self(None, tokens, attn_mask, output_attentions=True)
        logits = model_outputs.logits
        transformer_logs = get_transformer_logs(model_outputs.attentions, self.model, attn_mask)

        logs = {}
        predictions = logits[:, :-1, :]
        labels = tokens[:, 1:]
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        loss = F.cross_entropy(predictions, labels,
                               ignore_index=self.tokenizer.pad_token_id)
        n = (labels != self.tokenizer.pad_token_id).float().sum()
        logs['loss'] = (loss.item(), n)
        logs['transformer'] = transformer_logs
        return loss, logs, []
    
    def fetch_inputs_infer(self, scenes: List[Scene], dialogue_events_list: List[List[Event]]):
        tokens, attn_mask = self._tokenize_batch(dialogue_events_list, scenes)
        return None, tokens, attn_mask
    
    

class CustomerGPT2Evaluator(Evaluator):
    def __init__(self, max_generation_len=None, 
                 temp=1.0, top_k=None, top_p=None,
                 verbose=True) -> None:
        super(CustomerGPT2Evaluator, self).__init__()
        self.max_generation_len = max_generation_len
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.verbose = verbose
    
    def evaluate(self, model: CustomerGPT2, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        if self.verbose:
            predictions, _ = model.sample_full(scenes, [[]]*len(scenes), 
                                               max_generation_len=self.max_generation_len, 
                                               temp=self.temp, 
                                               top_k=self.top_k, 
                                               top_p=self.top_p)
            for prefix, prediction in predictions:
                prediction = prediction[0]
                print('======================')
                print(prefix)
                print(prediction)
                print('======================')
                print()
                print()
        return None

    

