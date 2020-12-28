# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import torch.utils.data as data
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from src.util.util import proj_root_dir
from src.args_and_config.args import args
from src.data_loader import load_train_validate_data,load_test_data, TrainDataset, ValidateDataset, TestDataset






def evaluate(model, xdata, ylabel):
    model.eval()

    with torch.no_grad():
        loss_function =nn.L1Loss()
        predicts = model(xdata)
        loss = loss_function(predicts.to(torch.float32), ylabel.to(torch.float32))
        return loss.item()












def main():
    test()


if __name__ == '__main__':
    main()