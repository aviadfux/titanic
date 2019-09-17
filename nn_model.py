import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(MLP, self).__init__()

        self.batch_size = batch_size
        INPUT_DIM = input_dim
        HIDDEN_DIM = [32, 16]

        self.linear = nn.Linear(INPUT_DIM, HIDDEN_DIM[0])
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM[0])
        self.dropout1 = nn.Dropout(0.5)

        self.hidden1 = nn.Linear(HIDDEN_DIM[0], HIDDEN_DIM[1])
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM[1])

        self.out = nn.Linear(HIDDEN_DIM[1], 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        output = self.linear(x)
        output = self.dropout1(output)
        #output = self.bn1(output)
        output = F.leaky_relu(output)

        output = self.hidden1(output)
        #output = self.bn2(output)
        output = F.leaky_relu(output)

        output = self.out(output)

        output = F.softmax(output)

        return output

    def predict(self, x):
        return np.argmax(self.forward(x).detach())