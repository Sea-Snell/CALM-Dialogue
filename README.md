# Context-Aware Language Modeling for Goal-Oriented Dialogue Systems

Official code for the paper "Context-Aware Language Modeling for Goal-Oriented Dialogue Systems"

[project site](https://sea-snell.github.io/CALM_LM_site/) | [arxiv]()

## **setup**

1. install requirements
2. `export PYTHONPATH="$PWD/offline_airdialogue"`
3. Download the processed data and model checkpoints [here](https://drive.google.com/drive/folders/1mnAGcgqyQC3ygILwwf-llxLf70nT9AT9?usp=sharing). The `outputs/` folder contains checkpoints for our main model, our task pretrained model, and our customer bot.

## **Training**
   *(Note: all training runs use wandb by default, you can turn off wandb syncing in the config.)*
* `cd scripts/train`
* To run data-parallel multi-GPU training, on any of the commands below replace `python <script_path>` with `python -m torch.distributed.launch --nproc_per_node <n_GPUs> --use_env <script_path>`.

* **Pretraining CALM**<br>
    *(two variants of the auxiliary loss function)*

    * 
        script: `python train_pretrain_table_agent.py`<br>
        config: `config/train_pretrain_table_agent.yaml`

    * 
        script: `python train_pretrain_simplified_aux_gpt2.py`<br>
        config: `config/train_pretrain_simplified_aux_gpt2.yaml`

* **Training the customer bot**

    * 
        script: `python train_customer.py`<br>
        config: `config/train_customer_bot.yaml`

* **Training CALM**<br>
    *(two variants of the auxiliary loss function)*

    * 
        script: `python train_real_table_agent.py`<br>
        config: `config/train_real_table_agent.yaml`

    * 
        script: `python train_simplified_aux_gpt2.py`<br>
        config: `config/train_simplified_aux_agent.yaml`

* **Training Standard LM**
    
    *
        script: `python train_basic_agent.py`<br>
        config: `config/train_basic_agent.yaml`

* **Training the reward model for Model Based Rollout Planning**

    *
        script: `python train_constraint_parser.py`<br>
        config: `config/train_constraint_parser.yaml`

## Evaluating

* `cd scripts/eval`

* **Simulated Evaluation**

    *
        script: `python selfplay_eval.py`<br>
        config: `config/selfplay_eval.yaml`<br><br>
        * A log of results will be saved to the location specified by `selfplay/outputs_file` in the config. To print out the success rate for the selfplay run: `python compute_results.py --results_file <your_eval_outputs_file>`<br><br>
        * Note: selfplay evaluation will by default use all the GPUs available on your machine. To Specify which GPUs to use, prefix the command with `CUDA_VISIBLE_DEVICES=<comma_seperated_list_of_gpu_indicies>`

* **Language Quality Evaluation**

    *
        script: `python language_quality_eval.py`<br>
        config: `config/language_eval.yaml`<br><br>
        * To parallelize evaluation across on multiple-GPUs, run: `python -m torch.distributed.launch --nproc_per_node <n_GPUs> --use_env language_quality_eval.py`
