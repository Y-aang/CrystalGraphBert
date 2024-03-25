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

from EnergyConstruct import MethodGraphBertEnergyConstruct
from utils import PerovskiteGBConfig


# data
input_x = []
input_y = []
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# TODO: Change the path to your data * 4
print("device:", device)
file_list = os.listdir("../../../result/WaterSplit_PGB/atom_padded/")

for cif_number in tqdm.tqdm(file_list):
    atom_file = "../../../result/WaterSplit_PGB/atom_padded/" + cif_number
    dis_file = "../../../result/WaterSplit_PGB/distance_padded/" + cif_number
    energy_file = "../../../result/WaterSplit_PGB/energy/" + cif_number
    with open(atom_file, "rb") as f_atom:
        atom = pickle.load(f_atom)
    with open(dis_file, "rb") as f_dis:
        distance = pickle.load(f_dis)
    with open(energy_file, "rb") as f_energy:
        energy = pickle.load(f_energy)
    input_x.append([atom, distance])
    input_y.append(energy)
input_x = torch.tensor(input_x, dtype=torch.float32)
input_y = torch.tensor(input_y)

customed_dataset = data.TensorDataset(input_x, input_y)
train_size = int(0.8 * len(customed_dataset))
test_size = len(customed_dataset) - train_size
train_dataset, test_dataset = data.random_split(customed_dataset, [train_size, test_size])

train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)





# parameters
np.random.seed(1)
torch.manual_seed(1)

nclass = 6
nfeature = 1000
ngraph = 19717

lr = 0.0005
k = 30
max_epoch = 200

#train
x_size = nfeature
hidden_size = intermediate_size = 128
num_attention_heads = 4
num_hidden_layers = 6
y_size = nclass
graph_size = ngraph
residual_type = 'graph_raw'

print('************ Start ************')

bert_config = PerovskiteGBConfig(residual_type = residual_type, k=80, x_size=1000, y_size=6, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
method_obj = MethodGraphBertEnergyConstruct(bert_config,device)  #MethodGraphBertConstantConstruct
method_obj.max_epoch = max_epoch
method_obj.lr = lr
method_obj.train_dataloader = train_loader
method_obj.test_dataloader = test_loader

method_obj.run(device=device)






print('************ Finish ************')