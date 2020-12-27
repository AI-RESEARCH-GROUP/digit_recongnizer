# -*- coding: UTF-8 -*-

import argparse

parser = argparse.ArgumentParser(description='GCN')
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--n-epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=True)")

parser.set_defaults(self_loop=True)
args = parser.parse_args()

print(args)
