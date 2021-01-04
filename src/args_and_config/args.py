# -*- coding: UTF-8 -*-

import argparse

parser = argparse.ArgumentParser(description='GCN')
parser.add_argument("--gpu", type=int, default=-1, help="gpu")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--n-epochs", type=int, default=10, help="number of training epochs")
args = parser.parse_args()

print(args)
