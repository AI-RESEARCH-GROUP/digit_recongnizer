# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from src.model.Net import Net
import torch.nn.functional as F
# from src.preprocess import data_into_tensors
from src.data_loader import MNIST_data
from torchvision import transforms


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

'''for get train_loader 
    from **********to******** 
'''
#***********
# Random Rotation Transformation
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageEnhance
class RandomRotation(object):
        """
        https://github.com/pytorch/vision/tree/master/torchvision/transforms
        Rotate the image by angle.
        Args:
            degrees (sequence or float or int): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
                An optional resampling filter.
                See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            expand (bool, optional): Optional expansion flag.
                If true, expands the output to make it large enough to hold the entire rotated image.
                If false or omitted, make the output image the same size as the input image.
                Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation.
                Origin is the upper left corner.
                Default is the center of the image.
        """

        def __init__(self, degrees, resample=False, expand=False, center=None):
            if isinstance(degrees, numbers.Number):
                if degrees < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                self.degrees = (-degrees, degrees)
            else:
                if len(degrees) != 2:
                    raise ValueError("If degrees is a sequence, it must be of len 2.")
                self.degrees = degrees

            self.resample = resample
            self.expand = expand
            self.center = center

        @staticmethod
        def get_params(degrees):
            """Get parameters for ``rotate`` for a random rotation.
            Returns:
                sequence: params to be passed to ``rotate`` for random rotation.
            """
            angle = np.random.uniform(degrees[0], degrees[1])

            return angle

        def __call__(self, img):
            """
                img (PIL Image): Image to be rotated.
            Returns:
                PIL Image: Rotated image.
            """

            def rotate(img, angle, resample=False, expand=False, center=None):
                """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)
                Args:
                img (PIL Image): PIL Image to be rotated.
                angle ({float, int}): In degrees degrees counter clockwise order.
                resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
                An optional resampling filter.
                See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
                expand (bool, optional): Optional expansion flag.
                If true, expands the output image to make it large enough to hold the entire rotated image.
                If false or omitted, make the output image the same size as the input image.
                Note that the expand flag assumes rotation around the center and no translation.
                center (2-tuple, optional): Optional center of rotation.
                Origin is the upper left corner.
                Default is the center of the image.
                """

                return img.rotate(angle, resample, expand, center)

            angle = self.get_params(self.degrees)

            return rotate(img, angle, self.resample, self.expand, self.center)
    # Random Vertical and Horizontal Shift'
class RandomShift(object):
        def __init__(self, shift):
            self.shift = shift

        @staticmethod
        def get_params(shift):
            """Get parameters for ``rotate`` for a random rotation.
            Returns:
                sequence: params to be passed to ``rotate`` for random rotation.
            """
            hshift, vshift = np.random.uniform(-shift, shift, size=2)

            return hshift, vshift

        def __call__(self, img):
            hshift, vshift = self.get_params(self.shift)

            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)

batch_size = 64

train_dataset = MNIST_data('data/train.csv', transform= transforms.Compose(
                            [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data('data/test.csv')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size, shuffle=False)
#***********
#Training and Evaluation

def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data[0]))

    def evaluate(data_loader):
        model.eval()
        loss = 0
        correct = 0

        for data, target in data_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            loss += F.cross_entropy(output, target, size_average=False).data[0]

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss /= len(data_loader.dataset)

        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

        #train the network
        n_epochs = 1
        for epoch in range(n_epochs):
            train(epoch)
            evaluate(train_loader)

def main():
    train(epoch=1)


if __name__ == "__main__":
    main()