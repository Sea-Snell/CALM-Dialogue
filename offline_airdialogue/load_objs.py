from ad.airdialogue import AirDialogue
from ad.synthetic_iterator import AirDialogueIterator
from ad.torch_datasets import BasicDataset, RealSynthTableDataset, RealSynthTableDataset2, ToySynthTableDataset, ToySynthTableDataset2
from bots.basic_policy_bot import BasicPolicyBot
from bots.full_policy_bot import FullPolicyBot
from bots.toy_customer_bot import ToyCustomerBot
from models.base import ConstraintRewardFunction, GPT2LMPolicy, LanguageEvaluator, OracleConstraintParser, OracleRewardFunction, PPLEvaluator, RemotePolicy
from models.basic_agent_gpt2 import BasicAgentGPT2, BasicAgentGPT2Evaluator
from models.constraint_predictor_roberta import ConstraintPredictorRoberta, ConstraintPredictorRobertaEvaluator
from models.customer_gpt2 import CustomerGPT2, CustomerGPT2Evaluator
from models.simplfied_aux_gpt2 import SimplifiedAuxGPT2Evaluator
from models.simplfied_aux_gpt2 import SimplifiedAuxGPT2
from models.table_agent_gpt2 import TableGPT2Agent, TableGPT2AgentEvaluator
from bots.mcts_bot import MCTSBot
from utils.data_utils import DiscreteFeatures
from utils.misc import convert_path
import torch

registry = {}

def register(name):
    def add_f(f):
        registry[name] = f
        return f
    return add_f

@register('discrete_features')
def initalize_features(config, verbose=True):
    return DiscreteFeatures.from_file(convert_path(config['path_to_discrete']), config['key'], config['strict'])


@register('constraint_parser_roberta')
def initalize_constraint_parser_roberta(config, device, verbose=True):
    discrete_features = load_item(config['discrete_features'], verbose=verbose)
    model = ConstraintPredictorRoberta(discrete_features=discrete_features, 
                                       roberta_type=config['roberta_type'], 
                                       device=device, 
                                       max_length=config['max_length']).to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print(f'loading constraint_predictor_roberta state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'))
        if verbose:
            print('loaded.')
    return model

@register('constraint_parser_roberta_evaluator')
def initalize_constraint_parser_roberta_evaluator(config, device, verbose=True):
    return ConstraintPredictorRobertaEvaluator(k=config['k'])

@register('table_gpt2_LM')
def initalize_table_gpt2_LM(config, device, verbose=True):
    discrete_features = load_item(config['discrete_features'], verbose=verbose)
    model = TableGPT2Agent(discrete_features=discrete_features, 
                           cond_reward_key=config['cond_reward_key'], 
                           gpt2_type=config['gpt2_type'],
                           device=device, 
                           max_length=config['max_length'], 
                           attn_prior=config['attn_prior'],
                           geometric_rate=config['geometric_rate']).to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print(f'loading table_gpt2_LM state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

@register('table_gpt2_evaluator')
def initalize_customer_gpt2_evaluator(config, device, verbose=True):
    opposing_bot = load_item(config['opposing_bot'], device, verbose=verbose)
    return TableGPT2AgentEvaluator(opposing_bot, 
                                   max_turns=config['max_turns'], 
                                   verbose=config['verbose'], 
                                   kind=config['kind'], 
                                   **config['generation_kwargs'])

@register('basic_agent_gpt2_LM')
def initalize_basic_agent_gpt2_LM(config, device, verbose=True):
    discrete_features = load_item(config['discrete_features'], verbose=verbose)
    model = BasicAgentGPT2(discrete_features=discrete_features, 
                           gpt2_type=config['gpt2_type'], 
                           device=device, 
                           max_length=config['max_length'], 
                           attn_prior=config['attn_prior'], 
                           geometric_rate=config['geometric_rate']).to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print(f'loading basic_agent_gpt2_LM state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

@register('basic_agent_gpt2_evaluator')
def initalize_basic_agent_gpt2_evaluator(config, device, verbose=True):
    return BasicAgentGPT2Evaluator(max_generation_len=config['max_generation_len'], 
                                   temp=config['temp'], 
                                   top_k=config['top_k'], 
                                   top_p=config['top_p'], 
                                   verbose=config['verbose'])

@register('simplified_aux_gpt2_LM')
def initalize_simplified_aux_gpt2_LM(config, device, verbose=True):
    discrete_features = load_item(config['discrete_features'], verbose=verbose)
    model = SimplifiedAuxGPT2(discrete_features=discrete_features, 
                              gpt2_type=config['gpt2_type'], 
                              device=device, 
                              max_length=config['max_length']).to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print(f'loading simplified_aux_gpt2_LM state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

@register('simplified_aux_gpt2_evaluator')
def initalize_simplified_aux_gpt2_evaluator(config, device, verbose=True):
    opposing_bot = load_item(config['opposing_bot'], device, verbose=verbose)
    return SimplifiedAuxGPT2Evaluator(opposing_bot, 
                                      max_turns=config['max_turns'], 
                                      verbose=config['verbose'], 
                                      kind=config['kind'], 
                                      **config['generation_kwargs'])

@register('customer_gpt2_LM')
def initalize_gpt2_customer_LM(config, device, verbose=True):
    model = CustomerGPT2(gpt2_type=config['gpt2_type'],
                         device=device,
                         max_length=config['max_length']).to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print(f'loading customer_gpt2 state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

@register('customer_gpt2_evaluator')
def initalize_customer_gpt2_evaluator(config, device, verbose=True):
    return CustomerGPT2Evaluator(max_generation_len=config['max_generation_len'], 
                                 temp=config['temp'], 
                                 top_k=config['top_k'], 
                                 top_p=config['top_p'], 
                                 verbose=config['verbose'])

@register('gpt2_lm_policy')
def initalize_gpt2_lm_sample_policy(config, device, verbose=True):
    lm = load_item(config['lm'], device, verbose=verbose)
    return GPT2LMPolicy(lm, config['kind'])

@register('remote_policy')
def initalize_remote_policy(config, device, verbose=True):
    return RemotePolicy(config['url'])

@register('language_evaluator')
def initalize_language_evaluator(config, deivce, verbose=True):
    return LanguageEvaluator(trun_first=config['trunc_first'], **config['generation_kwargs'])

@register('ppl_evaluator')
def initalize_ppl_evaluator(config, deivce, verbose=True):
    return PPLEvaluator()

@register('oracle_constraint_parser')
def initalize_oracle_constraint_parser(config, device, verbose=True):
    return OracleConstraintParser()

@register('oracle_reward')
def initalize_gpt2_lm_sample_policy(config, device, verbose=True):
    return OracleRewardFunction()

@register('constraint_reward')
def initalize_constraint_reward(config, device, verbose=True):
    constraint_parser = load_item(config['constraint_parser'], device, verbose=verbose)
    return ConstraintRewardFunction(constraint_parser)

@register('basic_policy_bot')
def initalize_basic_policy_bot(config, device, verbose=True):
    policy = load_item(config['policy'], device, verbose=verbose)
    return BasicPolicyBot(policy, **config['generation_kwargs'])

@register('full_policy_bot')
def initalize_basic_policy_bot(config, device, verbose=True):
    policy = load_item(config['policy'], device, verbose=verbose)
    return FullPolicyBot(policy, **config['generation_kwargs'])

@register('mcts_bot')
def initalize_basic_policy_bot(config, device, verbose=True):
    policy = load_item(config['policy'], device, verbose=verbose)
    reward = load_item(config['reward'], device, verbose=verbose)
    return MCTSBot(policy, reward, config['generation_kwargs'], config['reward_kwargs'])

@register('toy_customer_bot')
def initalize_toy_customer_bot(config, device, verbose=True):
    return ToyCustomerBot()

@register('airdialogue')
def initalize_airdialogue(config, verbose=True):
    return AirDialogue(convert_path(config['filepath']), 
                       limit=config['limit'], heauristic_filter=config['heauristic_filter'],
                       filter_with_goal=config['filter_with_goal'])

@register('synthetic_airdialogue')
def initalize_synthetic_airdialogue(config, verbose=True):
    return AirDialogueIterator(convert_path(config['ad_dir']))

@register('basic_dataset')
def initalize_basic_dataset(config, verbose=True):
    ad = load_item(config['data'], verbose=verbose)
    return BasicDataset(ad, 
                        cond_reward_key=config['cond_reward_key'], 
                        cond_reward=config['cond_reward'])

@register('toy_synth_table_dataset')
def initalize_toy_synth_table_dataset(config, verbose=True):
    ad = load_item(config['data'], verbose=verbose)
    return ToySynthTableDataset(ad, config['cond_reward_key'], 
                                reward_1_prob=config['reward_1_prob'])

@register('toy_synth_table_dataset2')
def initalize_toy_synth_table_dataset2(config, verbose=True):
    ad = load_item(config['data'], verbose=verbose)
    return ToySynthTableDataset2(ad, config['cond_reward_key'])

@register('real_synth_table_dataset')
def initalize_real_synth_table_dataset(config, verbose=True):
    real_ad = load_item(config['real_data'], verbose=verbose)
    synth_ad = load_item(config['synth_data'], verbose=verbose)
    return RealSynthTableDataset(synth_ad, real_ad, config['cond_reward_key'], 
                                 max_retries=config['max_retries'], 
                                 original_prob=config['original_prob'], 
                                 verbose=config['verbose'], 
                                 reward_1_prob=config['reward_1_prob'])

@register('real_synth_table_dataset2')
def initalize_real_synth_table_dataset2(config, verbose=True):
    real_ad = load_item(config['real_data'], verbose=verbose)
    synth_ad = load_item(config['synth_data'], verbose=verbose)
    return RealSynthTableDataset2(synth_ad, real_ad, config['cond_reward_key'], 
                                  max_retries=config['max_retries'], 
                                  original_prob=config['original_prob'], 
                                  verbose=config['verbose'])

def load_item(config, *args, verbose=True):
    config = config.copy()
    name = config.pop('name')
    if name not in registry:
        raise NotImplementedError
    if verbose:
        print(f'loading {name}: {config}')
    return registry[name](config, *args, verbose=verbose)
