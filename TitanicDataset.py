from __future__ import print_function, division
import pandas as pd
from torch.utils.data import Dataset
import torch

import features_extraction as fe

class TitanicDataset(Dataset):
    """Titanic dataset."""

    def __init__(self, data_set, is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with passenger data.
            train (bool): False is test, true(default) if train.
        """

        self.passengers = fe.extract(data_set, is_train)
        self.is_train = is_train

    def __len__(self):
        return len(self.passengers)

    def __getitem__(self, idx):
        if (self.is_train):
            features = torch.Tensor([list(self.passengers[idx][:-1])])
            target = torch.Tensor([(self.passengers[idx][-1])]).long()
            return features, target
        else:
            features = torch.Tensor([list(self.passengers[idx][:-1])])
            passenger_id = self.passengers[idx][-1]
            return features, passenger_id
