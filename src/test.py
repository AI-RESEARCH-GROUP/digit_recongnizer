# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import torch.utils.data as data
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from src.util.util import proj_root_dir,file_exists
from src.args_and_config.args import args
from src.data_loader import load_train_validate_data,load_test_data, TrainDataset, ValidateDataset, TestDataset

def evaluate(model, xdata, ylabel):
    model.eval()

    with torch.no_grad():
        loss_function =nn.L1Loss()
        predicts = model(xdata)
        loss = loss_function(predicts.to(torch.float32), ylabel.to(torch.float32))
        return loss.item()

def test():
    if not file_exists(proj_root_dir + 'checkpoints/model_parameters.pth'):
        print()
        print("please run train.py first !!!")
        print()
        exit(-1)
    # ================================================
    # 1) use cuda
    # ================================================
    cuda = True
    if args.gpu < 0:
        cuda = False

    # ================================================
    # 2) get data
    # ================================================

    train_validate = load_train_validate_data()
    test_data = load_test_data()

    # ================================================
    # 3) get model from checkpoints
    # ================================================
    model = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    model.load_state_dict(torch.load(proj_root_dir + 'checkpoints/model_parameters.pth'))
    if cuda:
        model.cuda()
    # ================================================
    # 4) eval/test
    # ================================================
    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    t0 = time.time()
    durations = []

    losses = []
    for step, (xdata_test_batch, ylabel_test_batch) in enumerate(test_dataloader):
        model.test()

        t0 = time.time()
        xdata_test = xdata_test_batch
        ylabel_test = ylabel_test_batch
        if cuda:
            xdata_test = xdata_test.to(args.gpu)
            ylabel_test = ylabel_test.to(args.gpu)

    loss = evaluate(model, xdata_test, ylabel_test)
    losses.append(loss)

    durations.append(time.time() - t0)
    print("Test loss {:.4f} | Time(s) {:.4f}s".format(np.mean(losses), np.mean(durations) / 1000))

def main():
    test()


if __name__ == '__main__':
    main()