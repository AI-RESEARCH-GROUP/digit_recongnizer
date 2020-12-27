# -*- coding: UTF-8 -*-

import numpy as np
from src.util.util import proj_root_dir ,file_exists
from src.args_and_config.config import config
from torch.utils import data
import math
from torch.utils.data.dataset import Dataset


#将训练集划分成训练集和验证集的比例函数，这里是7:3（可以在config.py里面修改比例）
def sample_split_ratio(train_validate_data):
    # train_validate_ratio = config["train_validate_ratio"].split(":")
    #
    # train_ratio = float(train_validate_ratio[0])/(float(train_validate_ratio[0])+float(train_validate_ratio[1]))
    # validate_ratio = float(1-train_ratio)
    #
    # n_train = math.floor(len(train_validate_data)*train_ratio)
    # n_train = n_train if n_train > 0 else 1
    # n_validate = math.floor(len(train_validate_data)*validate_ratio)
    # n_validate = n_train if n_validate > 0 else 1
    n_train = 30000
    n_validate = 12000
    return n_train , n_validate

#加载训练和验证集
def load_train_validate_data():
    if not file_exists(proj_root_dir + 'data/train_validate.npz'):
        print("data/train_validate.npz not exists, please run preprocess.py first !! ")
        exit(-1)

    train_validate = np.load(proj_root_dir + 'data/train_validate.npz')["train_set"]

    return train_validate

#加载测试集
def load_test_data():
    test = np.load(proj_root_dir + 'data/test.npz')["test_set"]
    return test

#获取训练、验证、测试集数量
train_validate = load_train_validate_data()
test = load_test_data()
n_train, n_validate = sample_split_ratio(train_validate)
# n_pixels =train_validate.shape[1]-1
n_test = len(test)


# class MyDataset(data.Dataset):
#
#     def __init__(self, ):
#         self.start_index = 0
# #todo there is a problem about how to distinguish between train data and test data
#         if n_pixels == 784:
#             # test data
#             self.X = data.iloc[:, 1:].values.astype('float')
#             self.y = None
#         else:
#             # training data
#             self.X = data.iloc[:, 1:].values.astype('float')
#             self.y = data['label'].values.astype('float')
#
#
#     def __getitem__(self, index):
#         if self.y is not None:
#             ylabel = data[self.start_index + index]
#             xdata = data[self.start_index + index]
#             return ylabel , xdata
#         else:
#             xdata = data[self.start_index + index]
#             return xdata

class TrainDataset(Dataset):
    def __init__(self, ):
        self.start_index = 0

    def __len__(self):
        return n_train

    def __getitem__(self, index):
        ylabel = train_validate[self.start_index + index][0]
        xdata = train_validate[self.start_index + index][1:]
        return xdata, ylabel


class ValidateDataset(Dataset):
    def __init__(self, ):
        self.start_index = n_train

    def __len__(self):
        return n_validate

    def __getitem__(self, index):
        ylabel = train_validate[self.start_index + index][0]
        xdata = train_validate[self.start_index + index][1:]
        return  xdata, ylabel


class TestDataset(Dataset):
    def __init__(self, ):
        self.start_index = 0

    def __len__(self):
        return n_test

    def __getitem__(self, index):
        xdata = train_validate[self.start_index + index]
        return xdata



def main():
    load_train_validate_data()
    load_test_data()

if __name__ == "__main__":
    main()
