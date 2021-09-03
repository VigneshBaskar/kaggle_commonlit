import os
import gc
import json
import torch
import shutil
import operator
import wandb
import numpy as np

from torch import nn
from pathlib import Path
from itertools import chain
from transformers import AutoModel
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup



def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()

class UnoTextDataset(Dataset):
    def __init__(self, text_excerpts, targets):
        self.text_excerpts = text_excerpts
        self.targets = targets
    
    def __len__(self):
        return len(self.text_excerpts)
    
    def __getitem__(self, idx):
        sample = {'text_excerpt': self.text_excerpts[idx],
                  'target': self.targets[idx]}
        return sample

def create_uno_text_dataloader(data, batch_size, shuffle, sampler, apply_preprocessing=True, num_workers=4, pin_memory=True, drop_last=False):
    # Preprocessing
    if apply_preprocessing:
        data['excerpt'] = data['excerpt'].apply(lambda x: x.replace('\n', ' '))
        data['excerpt'] = data['excerpt'].apply(lambda x: ' '.join(x.split()))
    
    text_excerpts = data['excerpt'].tolist()
    targets = data['target'].to_numpy().astype(np.float32).reshape(-1, 1)
    dataset = UnoTextDataset(text_excerpts=text_excerpts, targets=targets)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    return dataloader

class AttentionHead(nn.Module):
    def __init__(self, input_dim, head_hidden_dim):
        super(AttentionHead, self).__init__()
        head_hidden_dim = input_dim if head_hidden_dim is None else head_hidden_dim
        self.W = nn.Linear(input_dim, head_hidden_dim)
        self.V = nn.Linear(head_hidden_dim, 1)
        
    def forward(self, x):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class MaskFilledAttentionHead(nn.Module):
    def __init__(self, input_dim, head_hidden_dim):
        super(MaskFilledAttentionHead, self).__init__()
        head_hidden_dim = input_dim if head_hidden_dim is None else head_hidden_dim
        self.W = nn.Linear(input_dim, head_hidden_dim)
        self.V = nn.Linear(head_hidden_dim, 1)
        
    def forward(self, x, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores[attention_mask==0] = -10
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class MaskAddedAttentionHead(nn.Module):
    def __init__(self, input_dim, head_hidden_dim):
        super(MaskAddedAttentionHead, self).__init__()
        head_hidden_dim = input_dim if head_hidden_dim is None else head_hidden_dim
        self.W = nn.Linear(input_dim, head_hidden_dim)
        self.V = nn.Linear(head_hidden_dim, 1)
        
    def forward(self, x, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = attention_scores + attention_mask
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x

class RobertaPoolerOutputRegressor(nn.Module):
    def __init__(self, model_path, dropout_prob=0.1,  roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaPoolerOutputRegressor, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        pooler_output = roberta_outputs['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.regressor(pooler_output)
        return logits

class RobertaLastHiddenStateRegressor(nn.Module):
    def __init__(self, model_path, head_hidden_dim=None, dropout_prob=0.1, roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaLastHiddenStateRegressor, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.head = AttentionHead(input_dim=self.roberta.config.hidden_size, head_hidden_dim=head_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        last_hidden_state = roberta_outputs['last_hidden_state']
        attentive_vector = self.head(last_hidden_state)
        attentive_vector = self.dropout(attentive_vector)
        logits = self.regressor(attentive_vector)
        return logits

class RobertaMaskedLastHiddenStateRegressor(nn.Module):
    def __init__(self, model_path, head_hidden_dim=None, dropout_prob=0.1, roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaMaskedLastHiddenStateRegressor, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.head = AttentionHead(input_dim=self.roberta.config.hidden_size, head_hidden_dim=head_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        last_hidden_state = roberta_outputs['last_hidden_state']
        masked_last_hidden_state = last_hidden_state * torch.unsqueeze(inputs['attention_mask'], dim=2)
        attentive_vector = self.head(masked_last_hidden_state)
        attentive_vector = self.dropout(attentive_vector)
        logits = self.regressor(attentive_vector)
        return logits

class RobertaMaskFilledAttentionHeadRegressor(nn.Module):
    def __init__(self, model_path, dropout_prob=0.1, head_hidden_dim=None, roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaMaskFilledAttentionHeadRegressor, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.head = MaskFilledAttentionHead(input_dim=self.roberta.config.hidden_size, head_hidden_dim=head_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        last_hidden_state = roberta_outputs['last_hidden_state']
        attentive_vector = self.head(last_hidden_state, torch.unsqueeze(inputs['attention_mask'], dim=2))
        attentive_vector = self.dropout(attentive_vector)
        logits = self.regressor(attentive_vector)
        return logits

class RobertaMaskAddedAttentionHeadRegressor(nn.Module):
    def __init__(self, model_path, dropout_prob=0.1, head_hidden_dim=None, roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaMaskAddedAttentionHeadRegressor, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.head = MaskAddedAttentionHead(input_dim=self.roberta.config.hidden_size, head_hidden_dim=head_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        last_hidden_state = roberta_outputs['last_hidden_state']
        attentive_vector = self.head(last_hidden_state, torch.unsqueeze(inputs['attention_mask'], dim=2))
        attentive_vector = self.dropout(attentive_vector)
        logits = self.regressor(attentive_vector)
        return logits

class RobertaNHiddenStateRegressor(nn.Module):
    def __init__(self, model_path, dropout_prob=0.1, head_hidden_dim=None, roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaNHiddenStateRegressor, self).__init__()
        self.num_last_hidden_states = kwargs.pop('num_last_hidden_states')
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.head = MaskAddedAttentionHead(input_dim=self.roberta.config.hidden_size * self.num_last_hidden_states, head_hidden_dim=head_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size *  self.num_last_hidden_states, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs, output_hidden_states=True)
        last_hidden_states = torch.cat((roberta_outputs['hidden_states'][- self.num_last_hidden_states:]), axis=2)
        attentive_vector = self.head(last_hidden_states, torch.unsqueeze(inputs['attention_mask'], dim=2))
        attentive_vector = self.dropout(attentive_vector)
        logits = self.regressor(attentive_vector)
        return logits


class RobertaLastHiddenStateMeanPooler(nn.Module):
    def __init__(self, model_path, dropout_prob=0.1, head_hidden_dim=None,  roberta_hidden_dropout_prob=0.1, roberta_attention_probs_dropout_prob=0.1, **kwargs):
        super(RobertaLastHiddenStateMeanPooler, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_path,
                                                 hidden_dropout_prob=roberta_hidden_dropout_prob,
                                                 attention_probs_dropout_prob=roberta_attention_probs_dropout_prob, **kwargs)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, inputs):
        roberta_outputs = self.roberta(**inputs)
        last_hidden_state = roberta_outputs['last_hidden_state']
        masked_last_hidden_state = last_hidden_state * torch.unsqueeze(inputs['attention_mask'], dim=2)
        num_tokens = torch.unsqueeze(inputs['attention_mask'], dim=2)
        num_tokens = torch.clamp(num_tokens, min=1e-9)
        mean_embeddings = masked_last_hidden_state.sum(axis=1) / num_tokens.sum(axis=1)
        mean_embeddings = self.dropout(mean_embeddings)
        logits = self.regressor(mean_embeddings)
        return logits




def split_into_wd_groups(param_group, weight_decay):
    # Applies weight decay
    weight_parameters = {'params': [param_group['params'][index] for index, name in enumerate(param_group['param_names']) if 'weight' in name and 'LayerNorm' not in name],
                         'param_names': [param_group['param_names'][index] for index, name in enumerate(param_group['param_names']) if 'weight' in name and 'LayerNorm' not in name],
                         'lr': param_group['lr'],
                         'weight_decay': weight_decay,
                         'name': param_group['name']+'_weight'}
    # Does not apply weight decay
    bias_ln_parameters = {'params': [param_group['params'][index] for index, name in enumerate(param_group['param_names']) if 'bias' in name or 'LayerNorm' in name],
                          'param_names': [param_group['param_names'][index] for index, name in enumerate(param_group['param_names']) if 'bias' in name or 'LayerNorm' in name],
                          'lr': param_group['lr'],
                          'weight_decay': 0.0,
                          'name': param_group['name']+'_bias_ln'}
    parameters = [weight_parameters, bias_ln_parameters]
    return parameters


def get_optimizer_parameters(group_mode, lr, model, **kwargs):    
    non_bert = [(n,p) for n,p in model.named_parameters() if 'roberta' not in n]
    no_decay = ['bias', 'gamma', 'beta']

    if group_mode == 'i':
        optimizer_parameters = [{'params': [p for n,p in model.named_parameters() if ('bias' not in n) and ('pooler' not in n)],
                                 'lr': lr, 'weight_decay':0.01, 'name': 'weights'}, 
                                {'params': [p for n,p in model.named_parameters() if ('bias' in n) and ('pooler' not in n)],
                                 'lr': lr, 'weight_decay':0.00, 'name': 'bias'}]
        
    elif group_mode == 'j':
        optimizer_parameters = [{'params': [p for n,p in model.named_parameters() if ('bias' not in n)],
                                 'lr': lr, 'weight_decay':0.01, 'name': 'weights'}, 
                                {'params': [p for n,p in model.named_parameters() if ('bias' in n)],
                                 'lr': lr, 'weight_decay':0.00, 'name': 'bias'}]

    elif group_mode == 'k':
        # Finetuning task specific layers
        optimizer_parameters = [{'params': [p for n,p in non_bert if 'bias' not in n],
                                 'param_names': [n for n,p in non_bert if 'bias' not in n],
                                 'lr': lr, 'weight_decay':0.01, 'name': 'non_roberta_weights'}, 
                                {'params': [p for n,p in non_bert if 'bias' in n],
                                 'param_names': [n for n,p in non_bert if 'bias' in n],
                                 'lr': lr, 'weight_decay':0.00, 'name': 'non_roberta_bias'}]
        
    elif group_mode  == 's':
        multiplicative_factor = kwargs['multiplicative_factor']
        optimizer_parameters = [{'params': [p for n,p in model.roberta.named_parameters() if (not any(nd in n for nd in no_decay)) and ('pooler' not in n)],
                                 'lr': lr, 'weight_decay' : 0.01, 'name': 'roberta_weights'},
                                {'params': [p for n,p in model.roberta.named_parameters() if (any(nd in n for nd in no_decay))  and ('pooler' not in n)],
                                 'lr': lr,'weight_decay': 0.0, 'name': 'roberta_bias'},
                                {'params': [p for n,p in model.named_parameters() if all(nd not in n for nd in ['roberta','bias'])],
                                 'lr': lr * multiplicative_factor, 'weight_decay':0.01, 'name': 'non_roberta_weights'}, 
                                {'params': [p for n,p in non_bert if 'bias' in n],
                                 'lr': lr * multiplicative_factor, 'weight_decay':0.00, 'name': 'non_roberta_bias'}]
    elif group_mode == 'b':
        multiplicative_factor = kwargs['multiplicative_factor']
        model_parameters = {name: param for name, param in model.named_parameters()}
        model_parameters_names = [name for name, param in model.named_parameters()]
        
        # LR for task specific layers
        if kwargs['train_pooler']:
            task_specific_layer_names = [name for name, param in model.named_parameters() if 'regressor' in name or 'head' in name or 'pooler' in name or 'layer_norm' in name]
        else:
            task_specific_layer_names = [name for name, param in model.named_parameters() if 'regressor' in name or 'head' in name or 'layer_norm' in name]
        task_specific_layer_params = [model_parameters.get(name) for name in task_specific_layer_names]
        task_specific_optimizer_parameters = [{'params': task_specific_layer_params,
                                               'param_names': task_specific_layer_names,
                                               'lr': lr,
                                               'name': 'task_specific_layers'}]
        # LR for roberta layers
        # Freeze embeddings
        roberta_layer_names = [name for name, param in model.named_parameters() if 'roberta' in name]
        max_num_layers = model.roberta.config.num_hidden_layers
        roberta_layers_groups = {layer_num: {'params': [],
                                             'param_names': [],
                                             'lr': lr * multiplicative_factor ** (max_num_layers - layer_num),
                                             'name': f'layer_{layer_num}'} for layer_num in range(max_num_layers)}
        for layer_num in range(max_num_layers):
            for layer_name in roberta_layer_names:
                if f'layer.{layer_num}.' in layer_name:
                    roberta_layers_groups[layer_num]['param_names'].append(layer_name)
                    roberta_layers_groups[layer_num]['params'].append(model_parameters.get(layer_name))
        roberta_layers_optimizer_parameters = list(roberta_layers_groups.values())
        # Combine task specific layers and roberta layers
        optimizer_parameters = roberta_layers_optimizer_parameters + task_specific_optimizer_parameters

    elif group_mode == 'b_wd':
        multiplicative_factor = kwargs['multiplicative_factor']
        model_parameters = {name: param for name, param in model.named_parameters()}
        model_parameters_names = [name for name, param in model.named_parameters()]
        
        # LR for task specific layers
        if kwargs['train_pooler']:
            task_specific_layer_names = [name for name, param in model.named_parameters() if 'regressor' in name or 'head' in name or 'pooler' in name or 'layer_norm' in name]
        else:
            task_specific_layer_names = [name for name, param in model.named_parameters() if 'regressor' in name or 'head' in name or 'layer_norm' in name]
        task_specific_layer_params = [model_parameters.get(name) for name in task_specific_layer_names]
        task_specific_optimizer_parameters = [{'params': task_specific_layer_params,
                                               'param_names': task_specific_layer_names,
                                               'lr': lr,
                                               'name': 'task_specific_layers'}]
        # LR for roberta layers
        # Freeze embeddings
        roberta_layer_names = [name for name, param in model.named_parameters() if 'roberta' in name]
        max_num_layers = model.roberta.config.num_hidden_layers
        roberta_layers_groups = {layer_num: {'params': [],
                                             'param_names': [],
                                             'lr': lr * multiplicative_factor ** (max_num_layers - layer_num),
                                             'name': f'layer_{layer_num}'} for layer_num in range(max_num_layers)}
        for layer_num in range(max_num_layers):
            for layer_name in roberta_layer_names:
                if f'layer.{layer_num}.' in layer_name:
                    roberta_layers_groups[layer_num]['param_names'].append(layer_name)
                    roberta_layers_groups[layer_num]['params'].append(model_parameters.get(layer_name))
        roberta_layers_optimizer_parameters = list(roberta_layers_groups.values())
        # Combine task specific layers and roberta layers
        optimizer_parameters_without_wd = roberta_layers_optimizer_parameters + task_specific_optimizer_parameters

        # Assign weight decay
        weight_decay = kwargs['weight_decay']
        optimizer_parameters = []
        for layer_parameters in optimizer_parameters_without_wd:
            weight_parameters = {'params': [], 'param_names': [], 'lr': layer_parameters['lr'], 'name': layer_parameters['name']+'_weights', 'weight_decay': weight_decay}
            bias_parameters = {'params': [], 'param_names': [], 'lr': layer_parameters['lr'], 'name': layer_parameters['name']+'_bias', 'weight_decay': 0.0}
            layer_norm_parameters = {'params': [], 'param_names': [], 'lr': layer_parameters['lr'], 'name': layer_parameters['name']+'_layer_norm', 'weight_decay': 0.0}
            for param, param_name in zip(layer_parameters['params'], layer_parameters['param_names']):
                if 'LayerNorm' in param_name:
                    layer_norm_parameters['params'].append(param)
                    layer_norm_parameters['param_names'].append(param_name)
                elif 'bias' in param_name:
                    bias_parameters['params'].append(param)
                    bias_parameters['param_names'].append(param_name)
                else:
                    weight_parameters['params'].append(param)
                    weight_parameters['param_names'].append(param_name)
            optimizer_parameters.append(weight_parameters)
            optimizer_parameters.append(bias_parameters)
            optimizer_parameters.append(layer_norm_parameters)
            
    elif group_mode == 'be_wd':
        multiplicative_factor = kwargs['multiplicative_factor']
        weight_decay = kwargs['weight_decay']
        max_num_layers = model.roberta.config.num_hidden_layers

        # Task Specific Layer group
        tsl_param_group = [{'params': [param for name, param in model.named_parameters() if 'roberta' not in name],
                            'param_names': [name for name, param in model.named_parameters() if 'roberta' not in name],
                            'lr': lr,
                            'name': 'tsl'}]

        # Roberta Layer group
        roberta_layers_param_groups = []
        for layer_num in reversed(range(max_num_layers)):
            roberta_layer_param_groups = {'params': [param for name, param in model.named_parameters() if f'roberta.encoder.layer.{layer_num}.' in name],
                                          'param_names': [name for name, param in model.named_parameters() if f'roberta.encoder.layer.{layer_num}.' in name],
                                          'lr': lr * (multiplicative_factor ** (max_num_layers - layer_num)),
                                          'name': f'layer_{layer_num}'}
            roberta_layers_param_groups.append(roberta_layer_param_groups)

        # Embeddding group
        embedding_lr = lr * (multiplicative_factor ** (max_num_layers + 1))
        embedding_param_group = [{'params': [param for name, param in model.named_parameters() if 'embedding' in name],
                                  'param_names': [name for name, param in model.named_parameters() if 'embedding' in name],
                                  'lr': embedding_lr,
                                  'name': 'embedding'}]

        param_groups = tsl_param_group + roberta_layers_param_groups + embedding_param_group
        optimizer_parameters = list(chain(*[split_into_wd_groups(param_group, weight_decay=weight_decay) for param_group in param_groups]))

    elif group_mode == 'a':
        group1 = ['layer.0.','layer.1.','layer.2.','layer.3.']
        group2 = ['layer.4.','layer.5.','layer.6.','layer.7.']    
        group3 = ['layer.8.','layer.9.','layer.10.','layer.11.']
        group_all = group1 + group2 + group3
        optimizer_parameters = [
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': lr/2.6, 'name': 'roberta_group_1_weights'},
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': lr, 'name': 'roberta_group_2_weights'},
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': lr*2.6, 'name': 'roberta_group_3_weights'},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': lr/2.6, 'name': 'roberta_group_1_bias'},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': lr, 'name': 'roberta_group_2_bias'},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': lr*2.6, 'name': 'roberta_group_3_bias'},
                
            {'params': [p for n, p in non_bert if  'bias' not in n and 'head' in n], 'lr':lr*10, "momentum" : 0.99,'weight_decay_rate':0.01, 'name': 'head_weights'},
            {'params': [p for n, p in non_bert if  'bias' in n and 'head' in n], 'lr':lr*10, "momentum" : 0.99,'weight_decay_rate':0.0, 'name': 'head_bias'},

            {'params': [p for n, p in non_bert if  'bias' not in n and 'regressor' in n], 'lr':lr*10, "momentum" : 0.99,'weight_decay_rate':0.01, 'name': 'regressor_weights'},
            {'params': [p for n, p in non_bert if "bias"  in n and 'regressor' in n], 'lr':lr*10, "momentum" : 0.99,'weight_decay_rate':0.00, 'name': 'regressor_bias'},
        ]
    return optimizer_parameters

def make_cyclic_scheduler(optimizer, **kwargs):
    base_lr = kwargs['base_lr']
    step_size_up = kwargs['step_size_up']
    max_lr = kwargs['max_lr']
    cycle_momentum = kwargs['cycle_momentum']
    scheduler = CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=cycle_momentum)
    return scheduler

def make_cosine_schedule_with_warmup(optimizer, **kwargs):
    num_warmup_steps = kwargs['num_warmup_steps']
    num_training_steps = kwargs['num_training_steps']
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler

def make_linear_schedule_with_warmup(optimizer, **kwargs):
    num_warmup_steps = kwargs['num_warmup_steps']
    num_training_steps = kwargs['num_training_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler
    
def make_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kwargs):
    num_warmup_steps = kwargs['num_warmup_steps']
    num_training_steps = kwargs['num_training_steps']
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, 
                                                                   num_warmup_steps=num_warmup_steps,
                                                                   num_training_steps=num_training_steps,
                                                                   num_cycles=3)
    return scheduler

def make_polynomial_decay_schedule_with_warmup(optimizer, **kwargs):
    num_warmup_steps = kwargs['num_warmup_steps']
    num_training_steps = kwargs['num_training_steps']
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer, 
                                                          num_warmup_steps=num_warmup_steps,
                                                          num_training_steps=num_training_steps,
                                                          power=3.0)
    return scheduler
    
def get_scheduler(scheduler_type, optimizer, **kwargs):
    if scheduler_type == 'cyclic':
        scheduler = make_cyclic_scheduler(optimizer, **kwargs)
    elif scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = make_cosine_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'get_linear_schedule_with_warmup':
        scheduler = make_linear_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'cosine_with_hard_restarts_schedule_with_warmup':
        scheduler = make_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'polynomial_decay_schedule_with_warmup':
        scheduler = make_polynomial_decay_schedule_with_warmup(optimizer, **kwargs)
    else:
        scheduler = None
    return scheduler


def compute_predictions(text_excerpts, tokenizer, model, max_length, device, **kwargs):
    if max_length is None:
        # Sequence bucketing
        inputs = tokenizer(text=text_excerpts, padding=True, truncation=True, return_tensors='pt')
    else:
        inputs = tokenizer(text=text_excerpts, padding='max_length', truncation=True, max_length=max_length,  return_tensors='pt')
    inputs = {key:value.to(device) for key, value in inputs.items()}
    predictions = model(inputs)
    return predictions

def forward_pass_uno_text_batch(batch, tokenizer, model, compute_loss_fn, max_length, device, **kwargs):
    predictions = compute_predictions(text_excerpts=batch['text_excerpt'], tokenizer=tokenizer, model=model, max_length=max_length, device=device, **kwargs)
    outputs = {'predictions': predictions}
    loss = compute_loss_fn(outputs=outputs, targets=batch['target'], device=device, **kwargs) if compute_loss_fn is not None else None
    outputs['loss'] = loss
    return outputs


mse_loss_fn = nn.MSELoss()

def compute_mse_loss(outputs, targets, device, **kwargs):
    predictions = outputs['predictions']
    targets = targets.to(device)
    loss = mse_loss_fn(predictions, targets)
    return loss

def compute_rmse_loss(outputs, targets, device, **kwargs):
    predictions = outputs['predictions']
    targets = targets.to(device)
    loss = torch.sqrt(mse_loss_fn(predictions, targets))
    return loss

def compute_rmse_score(outputs, targets, **kwargs):
    predictions = outputs['predictions']
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    rmse_score = mean_squared_error(targets, predictions, squared=False)
    return rmse_score


class UnoStacker:
    def __init__(self):
        self.predictions = []
        self.targets = []
        
    def update(self, batch_outputs, batch_targets):
        self.predictions.append(batch_outputs['predictions'])
        self.targets.append(batch_targets)
    
    def get_stack(self):
        predictions = torch.vstack(self.predictions)
        targets = torch.vstack(self.targets)
        predictions = {'predictions': predictions}
        return predictions, targets

class Saver:
    def __init__(self, metric_name, is_lower_better, config, save_name, should_save=True):
        self.metric_name = metric_name
        self.save_path = Path(f'{save_name}/{metric_name}')
        self.operator_function = operator.le if is_lower_better else operator.ge
        self.best_score = np.inf if is_lower_better else 0
        self.best_iteration_num = 0
        self.config = config
        self.save_config()
        self.should_save = should_save
    
    def update(self, current_iteration_num, current_score, model, tokenizer):
        if self.operator_function(current_score, self.best_score):
            self.best_score = current_score
            self.best_iteration_num = current_iteration_num
            if self.should_save:
                self.save(model, tokenizer)
            print(f'{self.metric_name} attained best score: {current_score:.3f}. Saving the model')
            
    def save_config(self):
        shutil.rmtree(self.save_path, ignore_errors=True)
        os.makedirs(self.save_path)
        with open(self.save_path / 'config.json', 'w') as fp:
            json.dump(self.config, fp, sort_keys=True, indent=4)
    
    def save(self, model, tokenizer):
        shutil.rmtree(self.save_path, ignore_errors=True)
        os.makedirs(self.save_path)
        torch.save(model.state_dict(), self.save_path / 'model.pth')
        tokenizer.save_pretrained(self.save_path)
        
    def get_best_score(self):
        return {'best_score': self.best_score, 'best_iteration_num': self.best_iteration_num}


def train_one_batch(iteration_num, batch, tokenizer, model, optimizer, scheduler, forward_pass_fn, compute_loss_fn, max_length, accumulation_steps, device, **kwargs):
    model.train()
    if iteration_num == 0:
        optimizer.zero_grad()
    batch_loss = forward_pass_fn(batch=batch, tokenizer=tokenizer, model=model, 
                                 compute_loss_fn=compute_loss_fn, max_length=max_length, 
                                 device=device, **kwargs)['loss']                                  # Forward pass
    batch_loss = batch_loss / accumulation_steps                                                   # Normalize our loss (if averaged)
    batch_loss.backward()                                                                          # Backward pass
    if (iteration_num + 1) % accumulation_steps == 0:                                              # Wait for several backward steps
        optimizer.step()                                                                           # Now we can do an optimizer step
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    return model, batch_loss * accumulation_steps


def evaluate(dataloader, tokenizer, model, forward_pass_fn, compute_loss_fn, compute_metric_fn, stacker_class, max_length, device, **kwargs):
    epoch_loss = 0
    model.eval()
    stacker = stacker_class()
    with torch.no_grad():
        for batch_num, batch in enumerate(dataloader):
            batch_outputs = forward_pass_fn(batch=batch, tokenizer=tokenizer, model=model, 
                                            compute_loss_fn=compute_loss_fn, max_length=max_length, 
                                            device=device, **kwargs)
            batch_loss = batch_outputs['loss']
            epoch_loss += batch_loss.item()
            stacker.update(batch_outputs=batch_outputs, batch_targets=batch['target'])
    average_epoch_loss = epoch_loss/(batch_num+1)
    outputs, targets = stacker.get_stack()
    metric_score = compute_metric_fn(outputs=outputs, targets=targets, **kwargs)
    return average_epoch_loss, metric_score


def train_and_evaluate(num_epochs, train_dataloader, valid_dataloader, tokenizer, model, optimizer, scheduler,
                       forward_pass_fn_train, forward_pass_fn_valid, compute_loss_fn_train, compute_loss_fn_valid,
                       compute_metric_fn, stacker_class, max_length, accumulation_steps, validate_every_n_iteraion, 
                       valid_loss_saver, valid_score_saver, device, **kwargs):
    iteration_num = 0
    for epoch_num in range(num_epochs):
        for batch in train_dataloader:
            # for param_group in optimizer.param_groups:
            #     wandb.log({param_group['name']: {"lr": param_group['lr']}})
            model, iteration_train_loss = train_one_batch(iteration_num=iteration_num, batch=batch, tokenizer=tokenizer, model=model,
                                                          optimizer=optimizer, scheduler=scheduler, forward_pass_fn=forward_pass_fn_train,
                                                          compute_loss_fn=compute_loss_fn_train, max_length=max_length,
                                                          accumulation_steps=accumulation_steps, device=device, **kwargs)
            print(f'Epoch_num: {epoch_num}, iteration_num: {iteration_num}, iteration_train_loss: {iteration_train_loss}')
            # wandb.log({'Epoch_num': epoch_num, 'iteration_num': iteration_num, 'iteration_train_loss': iteration_train_loss})
            
            if 'validate_after_n_iteration' in kwargs:
                validate_after_n_iteration = kwargs['validate_after_n_iteration']
            else:
                validate_after_n_iteration = -1

            if ((iteration_num + 1) % validate_every_n_iteraion == 0) and (iteration_num > validate_after_n_iteration):
                valid_loss, valid_score = evaluate(dataloader=valid_dataloader, tokenizer=tokenizer, model=model,
                                                   forward_pass_fn=forward_pass_fn_valid, compute_loss_fn=compute_loss_fn_valid,
                                                   compute_metric_fn=compute_metric_fn, stacker_class=stacker_class,
                                                   max_length=max_length, device=device, **kwargs)
                valid_loss_saver.update(current_iteration_num=iteration_num,current_score=valid_loss, model=model, tokenizer=tokenizer)
                valid_score_saver.update(current_iteration_num=iteration_num, current_score=valid_score, model=model, tokenizer=tokenizer)
                # wandb.log({"iteration_num": iteration_num, "valid_loss": valid_loss, "valid_score": valid_score})
                print(f'Epoch_num: {epoch_num}, iteration_num: {iteration_num}, iteration_train_loss: {iteration_train_loss}')
                print(f'Epoch_num: {epoch_num}, iteration_num: {iteration_num}, valid_loss: {valid_loss}, valid_score: {valid_score}')
                # wandb.run.summary["best_valid_loss_iteration_num"] = valid_loss_saver.get_best_score()['best_iteration_num']
                # wandb.run.summary["best_valid_loss"] = valid_loss_saver.get_best_score()['best_score']
                # wandb.run.summary["best_valid_score_iteration_num"] = valid_score_saver.get_best_score()['best_iteration_num']
                # wandb.run.summary["best_valid_score"] = valid_score_saver.get_best_score()['best_score']
            iteration_num += 1
            
    if 'final_model_saver' in kwargs:
        final_model_saver = kwargs['final_model_saver']
        valid_score = 1
        final_model_saver.update(current_iteration_num=iteration_num, current_score=valid_score, model=model, tokenizer=tokenizer)
    
    # wandb.run.summary["best_valid_loss_iteration_num"] = valid_loss_saver.get_best_score()['best_iteration_num']
    # wandb.run.summary["best_valid_loss"] = valid_loss_saver.get_best_score()['best_score']
    # wandb.run.summary["best_valid_score_iteration_num"] = valid_score_saver.get_best_score()['best_iteration_num']
    # wandb.run.summary["best_valid_score"] = valid_score_saver.get_best_score()['best_score']

    output = {'model': model, 'best_score': valid_score_saver.get_best_score()['best_score'], 'best_loss': valid_loss_saver.get_best_score()['best_score']}
    return output
