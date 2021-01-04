# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from src.util.util import proj_root_dir

def preprocess():

    train_df = pd.read_csv(proj_root_dir + 'data/train.csv')
    test_df = pd.read_csv(proj_root_dir + 'data/test.csv')

    np.savez(proj_root_dir + 'data/train_validate.npz', train_set=train_df.values)
    np.savez(proj_root_dir + 'data/test.npz', test_set=test_df.values)


def main():
    preprocess()


if __name__ == "__main__":
    main()

