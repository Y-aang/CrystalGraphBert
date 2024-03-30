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
from tools import TensorBoardWriter

from MethodPerovskiteGB import MethodPerovskiteGB

# MethodGraphBertConstantConstruct
BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertEnergyConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''
    result_file_name = './result/result_water_split.txt'

    def __init__(self, config,device):
        super(MethodGraphBertEnergyConstruct, self).__init__(config)
        self.config = config
        self.bert = MethodPerovskiteGB(config)
        self.bert = self.bert.to(device)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.hidden_size*2)
        self.cls_y1 = torch.nn.Linear(config.hidden_size*2, 1)
        self.cls_y1 = self.cls_y1.to(device)
        self.cls_y = self.cls_y.to(device)
        self.init_weights()

    def forward(self, raw_features, dis_ids,device, idx=None):
        
        # print(raw_features.size(), dis_ids.size())
        outputs = self.bert(raw_features.to(device), dis_ids.to(device))
        # print(outputs)

        # sequence_output = 0
        sequence_output = torch.mean(outputs[0],dim=-2)
        # for i in range(self.config.k):
        #     sequence_output += outputs[0][:,i,:]
        # sequence_output /= float(self.config.k+1)
        # print(sequence_output)

        constant_hat = self.cls_y1(F.relu(self.cls_y(sequence_output)))
        
        # print(constant_hat)

        return constant_hat.squeeze(-1)


    def train_model(self, max_epoch, device):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        TensorBoard = TensorBoardWriter("./result/TensorBoard", accumulation_steps=1)
        global_steps = 0
        # for epoch in range(max_epoch):
        for epoch in range(max_epoch):  
            t_epoch_begin = time.time()

            # -------------------------
            # for step, (batch_x, batch_y) in enumerate(itertools.islice(self.dataloader, 10)):
            for step, (batch_x, batch_y) in enumerate(self.train_dataloader):
                self.train()
                optimizer.zero_grad()

                # print(step)
                # print(batch_x.size())
                # print(batch_y.size())
                output = self.forward(batch_x[:,0], batch_x[:,1],device=device)
                # print("output.size = ", output.size())
                # print("batch_y.size = ", batch_y.size())

                output = output.float().to(device)
                batch_y = batch_y.float().to(device)
                # loss_train = F.mse_loss(torch.tensor(output), batch_y)
                loss_train = F.l1_loss(output, batch_y)
                loss_train.requires_grad_(True)

                loss_train.backward()
                optimizer.step()

                self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}
                
                global_steps += 1
                if step % 50 == 0:
                    # test the model
                    relative_error, mae, loss_test = self.calculate_relative_error(device=device)
                    # mae = self.calculate_mae(device)
                    # print("accuracy =", accuracy)
                    relative_error = torch.norm(relative_error)
                    mae = torch.norm(mae)
                    # print("accuracy =", accuracy)

                    tensor_board_output = {
                        "Epoch" : epoch + 1,
                        "Train_loss" : loss_train.item(),
                        "Test_loss" : loss_test.item(),
                        "Relative_error" : relative_error,
                        "Mae" : mae,
                        "Time" : time.time() - t_epoch_begin
                    }
                    TensorBoard.write_dict(tensor_board_output, global_steps)
                    print('Epoch: {:04d}'.format(epoch + 1),
                      'Step: {:04d}'.format(step + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'relative_error: {:.4f}'.format(relative_error),
                      'Mae: {:.4f}'.format(mae),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))
                    
                    writein = 'Epoch: {:04d}'.format(epoch + 1)     \
                    + ', Step: {:04d}'.format(step + 1)   \
                    + ', loss_train: {:.4f}'.format(loss_train.item())    \
                    + ', relative_error: {:.4f}'.format(relative_error)    \
                    + ', Mae: {:.4f}'.format(mae)    \
                    + ', time: {:.4f}s'.format(time.time() - t_epoch_begin)   \
                    + "\n"
                    with open(self.result_file_name, 'a') as file:
                        file.write(writein)
                    
            # -------------------------
            # if epoch % 1 == 0:
            #     print('Epoch: {:04d}'.format(epoch + 1),
            #           'loss_train: {:.4f}'.format(loss_train.item()),
            #           'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self,device):

        with open(self.result_file_name, 'w') as file:
                        file.write('Train start:')
        self.train_model(self.max_epoch,device)

        return self.learning_record_dict
    
    def calculate_relative_error(self,device):
        total_relative_error = []
        total_mae = []
        total_loss = []
        self.eval()
        for step, (batch_x, batch_y) in enumerate(self.test_dataloader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = self.forward(batch_x[:,0], batch_x[:,1],device=device)
                # print(output)
                output = torch.tensor(output)
                # print('output = ', output)
                # print('batch_y = ', batch_y)

                # print("len accuracy = ", len(accuracy))
                batch_y = batch_y.to(device)
                total_loss.append(F.l1_loss(output, batch_y))
                relative_error = torch.mean(torch.abs(output - batch_y) / torch.abs(batch_y))
                mae = torch.mean(torch.abs(output - batch_y))
                # print('relative_error = ', relative_error)
                total_relative_error.append(relative_error)
                total_mae.append(mae)
        return sum(total_relative_error) / len(total_relative_error), sum(total_mae) / len(total_mae), sum(total_loss) / len(total_loss)
    
    def calculate_mae(self,device):
        accuracy = []
        self.eval()
        for step, (batch_x, batch_y) in enumerate(self.test_dataloader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = self.forward(batch_x[:,0], batch_x[:,1],device)
                # print(output)
                output = torch.tensor(output)
                # print('output = ', output)
                # print('batch_y = ', batch_y)

                # print("len accuracy = ", len(accuracy))
                
                relative_error = torch.abs(output - batch_y)
                # print('relative_error = ', relative_error)
                accuracy.append(relative_error)
        return sum(accuracy) / len(accuracy)