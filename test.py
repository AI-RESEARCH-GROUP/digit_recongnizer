# -*- coding: UTF-8 -*-

from src.test import main


if __name__ == '__main__':
    main()


# class MNISTCSVDataset(data.Dataset):
#
#     def __init__(self, csv_file, Train=False):
#         self.dataframe = pd.read_csv(csv_file, iterator=True)
#         self.Train = Train
#
#     def __len__(self):
#         if self.Train:
#             return 42000
#         else:
#             return 28000
#
#     def __getitem__(self, idx):
#         data = self.dataframe.get_chunk(100)
#         xdata = data.values.astype('float')
#         return xdata
#
#
# net = torch.load('checkpoints/divided-net.pkl')
#
# myMnist = MNISTCSVDataset(file)
# test_loader = torch.utils.data.DataLoader(myMnist, batch_size=1, shuffle=False)
#
# values = []
# for _, xd in enumerate(test_loader):
#     output = net(xd.squeeze().float())
#     values = values + output.argmax(dim=1).numpy().tolist()
#
# with open('data/sample_submission.csv', 'r') as fp_in, open('output/result.csv', 'w', newline='') as fp_out:
#     reader = csv.reader(fp_in)
#     writer = csv.writer(fp_out)
#     header = 0
#     for i, row in enumerate(reader):
#         if i == 0:
#             writer.writerow(row)
#         else:
#             row[-1] = str(values[i-1])
#             writer.writerow(row)
# 1.测试集数据npz,dataset dataloader,for 循环；
# 2.开启evl模式
# 3.加载最优参数放入model
# 4.cuda(输入和网络)
# 5.结束