import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class Seq2Frame(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, layer_num=2, dropout_rate=0.3, padding_index=None, device="cuda:0"):
        super(Seq2Frame, self).__init__()
        
        self.device = device
       
        if padding_index is not None:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_index)
        else:
            self.embedding  = nn.Embedding(vocab_size, hidden_size)

        self.lstm       = nn.LSTM(hidden_size, hidden_size, layer_num, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.linear     = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        vecs = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(vecs)
        prediction = self.linear(output[:, -1, :].squeeze(dim=1))

        return prediction

