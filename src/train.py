# -*- coding: UTF-8 -*-
import pandas as pd
import torch.utils.data as data
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from src.util.util import proj_root_dir
from src.args_and_config.args import args
from src.data_loader import load_train_validate_data,load_test_data, TrainDataset, ValidateDataset, TestDataset


def save_model_parameters(model):
    torch.save(model.state_dict(), proj_root_dir + 'checkpoints/model_parameters.pth')
    print("save model to %s " % (proj_root_dir + 'checkpoints/model_parameters.pth'))


def train():
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
    # 3) init model/loss/optimizer
    # ================================================

    model = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    if cuda:
        model.cuda()

    # ================================================
    # 4) model parameter init（初始化）
    # ================================================

    for param in model.parameters():
        print(param)
        nn.init.normal_(param, mean=0, std=0.01)
    loss_function = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)#lr=0.01,可在args中修改

    # ================================================
    # 5) train loop
    # ================================================
    #实例化
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    validate_dataset = ValidateDataset()
    validate_dataloader = DataLoader(validate_dataset, batch_size=10, shuffle=True)

    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    for epoch_i in range(0,args.n_epochs):
        print("Epoch {:05d} training...".format(epoch_i))
        durations = []
        for xdata_batch ,ylabel_batch in enumerate(train_dataloader):
            model.train()
            t0 = time.time()
            # =========================
            # get input parameter
            # =========================
            xdata = xdata_batch[100].int()
            ylabel = ylabel_batch[100].int()
            if cuda:
                xdata = xdata.to(args.gpu)
                ylabel = ylabel.to(args.gpu)
            #forward
            output = net(xdata.squeeze().float())
            ylabel = ylabel.long()
            loss = loss_function(output, ylabel.squeeze())
            #梯度清零
            optimizer.zero_grad()
            #做反向传播，看反向传播前后的误差
            loss.backward()
            #做优化，先要拿到梯度
            optimizer.step()#基于梯度，完成参数更新

            durations.append(time.time() - t0)

        # ================================================
        # 6) after each epochs ends
        # ================================================
        losses = []
        for xdata_batch_validate ,ylabel_batch_validate in enumerate(validate_dataloader):
            xdata_ = xdata_batch_validate[100].int()
            ylabel = ylabel_batch_validate[100].int()

#定义神经网络
net = nn.Sequential(
    nn.Linear(28 * 28, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
#ifcuda。。。。。
#todo 参数初始化没有
# s损失函数、优化器
loss_function = nn.MultiMarginLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# for epoch_i in range(0, args.n_epochs):
#     print("Epoch {:05d} training...".format(epoch_i))
#     durations = []
#
#     for g_adj_batch, features_batch, labels_batch in train_dataloader:


    for step, (yl, xd) in enumerate(train_loader):
        # model.train()
        # if cuda:....

        output = net(xd.squeeze().float())
        yl = yl.long()
        loss = loss_function(output, yl.squeeze())
        optimizer.zero_grad()
        #做反向传播，看反向传播前后的误差
        loss.backward()
        #做优化，先要拿到梯度
        optimizer.step()#基于梯度，完成参数更新

    if step % 20 == 0:
        print('step %d loss %.3f' % (step, loss))

torch.save(net, 'checkpoints/divided-net.pkl')
print("The results are saved in'checkpoints/divided-net.pkl' ")
#1.缺少迭代次数epoch
#2.缺少main（），将源代码放入src中
# 没有cuda判断
#没有从npz加载数据
#缺少测试、缺少训练模式和测试模式的开启
#enumerate






def main():
    train()


if __name__ == "__main__":
    main()