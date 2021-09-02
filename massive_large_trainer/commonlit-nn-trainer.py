#!/usr/bin/env python
# coding: utf-8

# # Objective
# The objective of this notebook is to train a five fold model with the best hyper-parameter settings that we are aware of for now. In the next step we shall re-run this notebook for the full dataset and compare the results to see if training on full dataset has resulted in a jump. 
# 
# The current known best settings are:
# 1. optimizer group: `be_wd`
# 2. learning rate: `3e-5`
# 3. multiplicative factor: `0.925`
# 4. weight decay: `0.02`
# 5. scheduler: `cosine_annealing_with_warmup`
# 6. model: `RobertaMaskAddedAttentionHeadRegressor` 
# 7. accumulation_steps: 1
# 8. All dropout_probs: 0 
# 9. num_epochs: 3
# 
# Make sure this notebook can train on our usual cross validatin strategy as well as full data seamlessly.

# In[1]:


import os
import wandb
import torch
import pandas as pd

from kaggle_secrets import UserSecretsClient
from transformers import AdamW, AutoTokenizer
from commonlit_nn_kit import seed_everything, clear_cuda, forward_pass_uno_text_batch
from commonlit_nn_kit import get_scheduler, get_optimizer_parameters, create_uno_text_dataloader, get_linear_schedule_with_warmup
from commonlit_nn_kit import compute_mse_loss, compute_rmse_loss, compute_rmse_score, train_and_evaluate, Saver, UnoStacker, RobertaMaskAddedAttentionHeadRegressor, RobertaLastHiddenStateMeanPooler
os.environ["WANDB_API_KEY"] = '72dd109a7a2f5f8fb4b0f8f15019d9f7ab550da7'


# In[4]:


experiment_parameters = []

count = 0
model_size = 'large'

for random_state in [42]:
    for base_model_path in [f'../input/robertas/roberta-{model_size}', '../input/commonlitmlm/mlm_competition_data']:
        for train_on_full_data in [False, True]:
            for model_class in [RobertaLastHiddenStateMeanPooler, RobertaMaskAddedAttentionHeadRegressor]:
                if train_on_full_data:
                    training_data_file_name = 'stratified_simple'
                    experiment_parameters.append({'name': f'exp_{count}',
                                                  'random_state': random_state,
                                                  'train_on_full_data': train_on_full_data,
                                                  'base_model_path': base_model_path,
                                                  'model_class': model_class,
                                                  'training_data_file_name': training_data_file_name})
                    count += 1
                else:
                    for training_data_file_name in ['kfold_simple', 'stratified_simple']:
                        experiment_parameters.append({'name': f'exp_{count}',
                                                      'random_state': random_state,
                                                      'train_on_full_data': train_on_full_data,
                                                      'base_model_path': base_model_path,
                                                      'model_class': model_class,
                                                      'training_data_file_name': training_data_file_name})
                        count += 1
                        
for random_state in [1000]:
    for base_model_path in [f'../input/robertas/roberta-{model_size}', '../input/commonlitmlm/mlm_competition_data']:
        for train_on_full_data in [False, True]:
            for model_class in [RobertaLastHiddenStateMeanPooler, RobertaMaskAddedAttentionHeadRegressor]:
                if train_on_full_data:
                    training_data_file_name = 'stratified_simple'
                    experiment_parameters.append({'name': f'exp_{count}',
                                                  'random_state': random_state,
                                                  'train_on_full_data': train_on_full_data,
                                                  'base_model_path': base_model_path,
                                                  'model_class': model_class,
                                                  'training_data_file_name': training_data_file_name})
                    count += 1
                else:
                    for training_data_file_name in ['kfold_simple', 'stratified_simple']:
                        experiment_parameters.append({'name': f'exp_{count}',
                                                      'random_state': random_state,
                                                      'train_on_full_data': train_on_full_data,
                                                      'base_model_path': base_model_path,
                                                      'model_class': model_class,
                                                      'training_data_file_name': training_data_file_name})
                        count += 1

pd.DataFrame(experiment_parameters).to_csv('experiment_parameters.csv')
experiment_parameters = experiment_parameters[11:]
experiment_parameters = [experiment_parameter for experiment_parameter in experiment_parameters if experiment_parameter['train_on_full_data']]


# In[3]:


num_folds = 5

for experiment_parameter in experiment_parameters:
    config = {}
    config.update(experiment_parameter)
    config['model_class'] = experiment_parameter['model_class'].__name__
    model_class = experiment_parameter['model_class']
    
    for fold in range(num_folds):
        config['save_name'] = f"experiments/{experiment_parameter['name']}/fold_{fold}"
        config['run_name'] = f"{experiment_parameter['name']}_fold_{fold}"

        config.update({
            'train_on_sample': False,
            'fold': fold, 
            'apply_preprocessing': False,
            'batch_size': 8,
            'num_epochs': 3,
            'validate_every_n_iteration': 1000 if config['train_on_full_data'] else 10,
            'validate_after_n_iteration': 1000 if config['train_on_full_data'] else 500,
            'tokenizer_name': f'../input/robertas/roberta-{model_size}',
            'dropout_prob': 0.0,
            'roberta_hidden_dropout_prob': 0.0,
            'roberta_attention_probs_dropout_prob': 0.0,
            'layer_norm_eps': 1e-7,
            'head_hidden_dim': 512,
            'group_mode': 'be_wd',
            'lr': 3e-5,
            'multiplicative_factor': 0.925,
            'eps': 1e-7,
            'weight_decay': 0.02,
            'scheduler_type': 'cosine_schedule_with_warmup',
            'num_warmup_steps': 0,
            'should_save_best_valid_loss_model': False,
            'should_save_best_valid_score_model': False if config['train_on_full_data'] else True,
            'should_save_final_model': True,
            'max_length': 256,
            'accumulation_steps': 1})


        if config['train_on_sample']:
            config.update({'num_epochs': 2,
                           'sample_size': 7,
                           'batch_size': 4,
                           'validate_every_n_iteration': 1,
                           'validate_after_n_iteration': -1})

        seed_everything(seed=config['random_state']+config['fold'])

        data = pd.read_csv(f"../input/commonlit-splits/commonlittrain_{config['training_data_file_name']}.csv")

        if config['train_on_full_data']:
            data = data.sample(frac=1, random_state=config['random_state'] + config['fold'])
            train_data, valid_data = data[:-2], data[-2:]
        else:
            train_data, valid_data = data[data['fold']!=config['fold']], data[data['fold']==config['fold']]

        if config['train_on_sample']:
            train_data = train_data[:config['sample_size']]
            valid_data = valid_data[:config['sample_size']]

        print(f'Length of train data: {len(train_data)}')
        print(f'Length of valid data: {len(valid_data)}')

        train_dataloader = create_uno_text_dataloader(data=train_data, batch_size=config['batch_size'], shuffle=True, sampler=None, apply_preprocessing=config['apply_preprocessing'])
        valid_dataloader = create_uno_text_dataloader(data=valid_data, batch_size=config['batch_size'], shuffle=False, sampler=None, apply_preprocessing=config['apply_preprocessing'])

        print(f'Number of batches in the train_dataloader: {len(train_dataloader)}')
        print(f'Number of batches in the valid_dataloader: {len(valid_dataloader)}')

        clear_cuda()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['tokenizer_name'])
        model = model_class(model_path=config['base_model_path'],
                            head_hidden_dim=config['head_hidden_dim'],
                            dropout_prob=config['dropout_prob'],
                            roberta_hidden_dropout_prob=config['roberta_hidden_dropout_prob'],
                            roberta_attention_probs_dropout_prob=config['roberta_attention_probs_dropout_prob'],
                            layer_norm_eps=config['layer_norm_eps'])
        _ = model.to(device)

        optimizer_parameters = get_optimizer_parameters(group_mode=config['group_mode'], lr=config['lr'],
                                                        model=model, multiplicative_factor=config['multiplicative_factor'], 
                                                        weight_decay=config['weight_decay'])
        optimizer = AdamW(optimizer_parameters, eps=config['eps'])

        scheduler = get_scheduler(scheduler_type=config['scheduler_type'], optimizer=optimizer,
                                  num_warmup_steps=config['num_warmup_steps'],
                                  num_training_steps=config['num_epochs'] * len(train_dataloader))

        valid_loss_saver = Saver(metric_name='rmse_loss', is_lower_better=True, config=config, save_name=config['save_name'], should_save=config['should_save_best_valid_loss_model'])
        valid_score_saver = Saver(metric_name='rmse_score', is_lower_better=True, config=config, save_name=config['save_name'], should_save=config['should_save_best_valid_score_model'])
        final_model_saver = Saver(metric_name='final_model', is_lower_better=True, config=config, save_name=config['save_name'], should_save=config['should_save_final_model'])

        run  = wandb.init(reinit=True,
                    project=f"commonlit-massive-training-{model_size}",
                    config=config,
                )
        wandb.run.name = config['run_name']

        with run:
            _ = train_and_evaluate(num_epochs=config['num_epochs'], train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, tokenizer=tokenizer,
                                   model=model, optimizer=optimizer, scheduler=scheduler,
                                   forward_pass_fn_train=forward_pass_uno_text_batch, forward_pass_fn_valid=forward_pass_uno_text_batch,
                                   compute_loss_fn_train=compute_mse_loss, compute_loss_fn_valid=compute_rmse_loss,
                                   compute_metric_fn=compute_rmse_score, stacker_class=UnoStacker,
                                   max_length=config['max_length'], accumulation_steps=config['accumulation_steps'],
                                   validate_every_n_iteraion=config['validate_every_n_iteration'], validate_after_n_iteration=config['validate_after_n_iteration'],
                                   valid_loss_saver=valid_loss_saver, valid_score_saver=valid_score_saver, final_model_saver=final_model_saver, device=device)
            clear_cuda()


# In[ ]:




