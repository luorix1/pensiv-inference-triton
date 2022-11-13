import os

import numpy as np
import torch
import torch.nn as nn

from .basic_block import BasicBlock
from .resnet import resnet18


class CNNModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNNModel, self).__init__()

        # Vision Model
        self.vision = resnet18(in_channels=in_channels, num_classes=num_classes)

    def forward(self, img):
        """img.shape = bn,3l,h,w, img_feature.shape = bn,512"""
        img_feature = self.vision(img)

        return img_feature


class LSTMModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_lstm_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.rnn = nn.LSTM(
            feature_dim, hidden_dim, n_lstm_layers, batch_first=True, bidirectional=True
        )
        self.rnn.flatten_parameters()
        self.output = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img_feature):
        h = torch.zeros(
            (2 * self.n_lstm_layers, img_feature.shape[0], self.hidden_dim),
            device=img_feature.device,
        )
        h = (h, h)
        img_feature = self.dropout(img_feature)
        img_feature, _ = self.rnn(img_feature, h)
        output = self.output(img_feature)
        return output
