import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, layer_num=2, dropout_rate=0.3, input_dropout_rate=0.5, device="cuda"):
        super(AutoEncoder, self).__init__()
        self.device = device
        self.linear_input = nn.Linear(vocab_size, hidden_size).to(self.device)
        self.linear_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size).to(self.device) for _ in range(layer_num)]
        )
        self.linear_output = nn.Linear(hidden_size, output_size).to(self.device)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_dropout = nn.Dropout(input_dropout_rate)

    def forward(self, x):
        # input layer
        x = x.to(self.device)
        x = self.input_dropout(x)
        x = self.dropout(F.leaky_relu(self.linear_input(x)))

        # hidden layer
        for layer in self.linear_list:
            x = self.dropout(F.leaky_relu(layer(x)) + x)
       
        # output layer
        x = F.relu(self.linear_output(x))

        return x

