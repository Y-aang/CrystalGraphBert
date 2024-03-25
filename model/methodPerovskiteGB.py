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

from utils import BertEmbeddings, BertEncoder


BertLayerNorm = torch.nn.LayerNorm

class MethodPerovskiteGB(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(MethodPerovskiteGB, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, raw_features, dis_ids, head_mask=None, residual_h=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(raw_features=raw_features, dis_ids=dis_ids)
        # print(embedding_output)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def run(self):
        pass