# -*- coding: UTF-8 -*-
import numpy as np
import time
import torch
from torch.utils.data.dataloader import DataLoader
from src.model.Sequential import Sequential
from src.util.util import proj_root_dir,file_exists
from src.args_and_config.args import args
from src.data_loader import  PredictDataset

def predicts():
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
    # 3) get model from checkpoints
    # ================================================
    model = Sequential()
    model.load_state_dict(torch.load(proj_root_dir + 'checkpoints/model_parameters.pth'))
    if cuda:
        model.to(args.gpu)
    # ================================================
    # 4) eval/test
    # ================================================
    predict_dataset = PredictDataset()
    predict_dataloader = DataLoader(predict_dataset, batch_size=10, shuffle=True)

    t0 = time.time()
    durations = []

    for xdata_test_batch in predict_dataloader:

        t0 = time.time()
        xdata_test = xdata_test_batch

        if cuda:
            xdata_test = xdata_test.to(args.gpu)

    predicts_result = model(xdata_test.float())
    print(predicts_result)

    durations.append(time.time() - t0)
    print(" Time(s) {:.4f}s". format(np.mean(durations) / 1000))

def main():
    predicts()


if __name__ == '__main__':
    main()