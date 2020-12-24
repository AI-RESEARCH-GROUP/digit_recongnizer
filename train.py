# -*- coding: UTF-8 -*-
import pandas as pd
import torch.utils.data as data
import torch
import torch.nn as nn

file = 'data/train.csv'
LR = 0.01


class MNISTCSVDataset(data.Dataset):

    def __init__(self, csv_file, Train=True):
        self.dataframe = pd.read_csv(csv_file, iterator=True)
        self.Train = Train
#TODO 没有从训练集中划分训练和验证集
# 长度动态计算
    def __len__(self):
        if self.Train:
            return 42000
        else:
            return 28000
#todo
    def __getitem__(self, idx):
        data = self.dataframe.get_chunk(idx)
        ylabel = data['label'].values.astype('float')
        xdata = data.iloc[:, 1:].values.astype('float')
        return ylabel, xdata

#训练实例化
mydataset = MNISTCSVDataset(file)

train_loader = torch.utils.data.DataLoader(mydataset, batch_size=1, shuffle=True)

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