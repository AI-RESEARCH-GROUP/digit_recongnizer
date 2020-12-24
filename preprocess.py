# -*- coding: UTF-8 -*-
import pandas as pd


def preprocess():
    # 加载训练数据
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

#processs需要做的：
# 1.把csv文件转成dataframe格式，再转成numpy格式
# 2.处理有问题的数据
# 3.保存npz格式

def main():
    preprocess()


if __name__ == "__main__":
    main()
