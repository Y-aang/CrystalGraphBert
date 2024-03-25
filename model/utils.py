import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# from transformers.modeling_bert import BertPredictionHeadTransform, BertAttention, BertIntermediate, BertOutput
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertAttention, BertIntermediate, BertOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
import os
import pickle
import time
import numpy as np
import tqdm
import itertools


BertLayerNorm = torch.nn.LayerNorm

class PerovskiteGBConfig(PretrainedConfig):

    def __init__(
        self,
        residual_type = 'none',
        x_size=1000,
        y_size=6,
        k=80,
        max_wl_role_index = 100,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        max_atom_index = 250,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        **kwargs
    ):
        super(PerovskiteGBConfig, self).__init__(**kwargs)
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.max_atom_index = max_atom_index
        self.residual_type = residual_type
        self.x_size = x_size
        self.y_size = y_size
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, residual_h=None):
        all_hidden_states = ()
        all_attentions = ()
        # hidden_states = hidden_states.unsqueeze(0)
        
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            #---- add residual ----
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:,index,:] += residual_h

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from features, wl, position and hop vectors.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.raw_feature_embeddings = nn.Embedding(config.max_atom_index, config.hidden_size, padding_idx=0)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_features=None, dis_ids=None):
        
        # print("BertEmbeddings：raw_features", raw_features.size())
        
        # print("raw_features type:", type(raw_features))
        raw_feature_embeds = self.raw_feature_embeddings(raw_features.int())
        # embeddings = []
        # for atom, dis in zip(raw_feature_embeds, dis_ids):
        #     centre_atom = atom[0]
            
        #     for i in range(1, len(atom)):
        #         centre_atom = centre_atom + atom[i] / dis[i]
        #         # if torch.isnan(centre_atom).any().item():
        #         #     print(centre_atom)
        #         # print(centre_atom)
        #     embeddings.append(centre_atom.tolist())
        dis_ids = dis_ids.unsqueeze(-1)
        # if torch.isnan(dis_ids).any().item():
        #     print('before problem')
        # if torch.isnan(raw_feature_embeds).any().item():
        #     print('before problem')
        embeddings = raw_feature_embeds + dis_ids
        embeddings = embeddings.sum(dim=-2)

        # embeddings = torch.tensor(embeddings)
        # if torch.isnan(embeddings).any().item():
        #     print('after problem')
        embeddings = self.LayerNorm(embeddings.float())
        embeddings = self.dropout(embeddings)
        return embeddings

class NodeConstructOutputLayer(nn.Module):
    def __init__(self, config):
        super(NodeConstructOutputLayer, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.x_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.x_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs