# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms
import numbers
from PIL import Image, ImageOps, ImageEnhance
from src.data_loader import MNIST_data



def preprocess():
    # 加载训练数据
    train_df = pd.read_csv('data/train.csv')

    n_train = len(train_df)
    n_pixels = len(train_df.columns) - 1
    n_class = len(set(train_df['label']))

    print('Number of training samples: {0}'.format(n_train))
    print('Number of training pixels: {0}'.format(n_pixels))
    print('Number of classes: {0}'.format(n_class))
    test_df = pd.read_csv('data/test.csv')

    # 加载测试数据
    n_test = len(test_df)
    n_pixels = len(test_df.columns)

    print('Number of train samples: {0}'.format(n_test))
    print('Number of test pixels: {0}'.format(n_pixels))

    #展示一些数据
    random_sel = np.random.randint(n_train, size=8)

    # grid = make_grid(
    #   torch.Tensor((train_df.iloc[random_sel, 1:].as_matrix() / 255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
    grid = make_grid(
        torch.Tensor((train_df.iloc[random_sel, 1:].values / 255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)

    plt.rcParams['figure.figsize'] = (16, 2)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    print(*list(train_df.iloc[random_sel, 0].values), sep=', ')
    # print("##########")
    plt.show()
    #Histogram of the classes
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
    plt.xticks(np.arange(n_class))
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.grid('on', axis='y')
    # print("*****************************")
    plt.show()
    # Random Rotation Transformation
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

    # Visualize the Transformations
    rotate = RandomRotation(20)
    shift = RandomShift(3)
    composed = transforms.Compose([RandomRotation(20),
                                   RandomShift(3)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = transforms.ToPILImage()(train_df.iloc[65, 1:].values.reshape((28, 28)).astype(np.uint8)[:, :, None])

    for i, tsfrm in enumerate([rotate, shift, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')

    plt.show()

    #load the Data into Tensors
    def data_into_tensors():
        batch_size = 64

        train_dataset = MNIST_data('data/train.csv', transform= transforms.Compose(
                                    [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
                                     transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
        test_dataset = MNIST_data('data/test.csv')


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size, shuffle=False)

        return train_loader,test_loader
def main():
    preprocess()


if __name__ == "__main__":
    main()
