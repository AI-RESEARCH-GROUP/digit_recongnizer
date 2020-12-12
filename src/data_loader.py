# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNIST_data(Dataset):
    """MNIST data set"""

    def __init__(self, file_path,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):

        df = pd.read_csv(file_path)

        if len(df.columns) == 784:
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])

def main():
    MNIST_data(Dataset)

if __name__ == "__main__":
    main()