# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Dropout
from torch.nn import ReLU, Softmax
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")

class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shpae of x : [batch,channel,axis1]
        x_real = x[:, 0:x.shape[1]//2, :]
        x_img = x[:, x.shape[1] // 2 : x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        output = torch.cat((real, imaginary), dim=1)
        return output

class CVCNN(nn.Module):
    def __init__(self,num_classes=11,adsb_is=False):
        super(CVCNN, self).__init__()
        self.adsb_is = adsb_is
        self.conv1 = ComplexConv(in_channels=1, out_channels=64, kernel_size=3)
        self.batchnorm1 = BatchNorm1d(num_features=128)
        self.maxpool1 = MaxPool1d(kernel_size=2)

        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=128)
        self.maxpool2 = MaxPool1d(kernel_size=2)

        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=128)
        self.maxpool3 = MaxPool1d(kernel_size=2)

        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=128)
        self.maxpool4 = MaxPool1d(kernel_size=2)

        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=128)
        self.maxpool5 = MaxPool1d(kernel_size=2)

        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=128)
        self.maxpool6 = MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=128)
        self.maxpool7 = MaxPool1d(kernel_size=2)

        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=128)
        self.maxpool8 = MaxPool1d(kernel_size=2)

        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=128)
        self.maxpool9 = MaxPool1d(kernel_size=2)

        self.flatten = Flatten()
        self.linear1 = LazyLinear(1024)
        self.linear2 = LazyLinear(num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)
        if self.adsb_is:
            x = self.conv6(x)
            x = F.relu(x)
            x = self.batchnorm6(x)
            x = self.maxpool6(x)

            x = self.conv7(x)
            x = F.relu(x)
            x = self.batchnorm7(x)
            x = self.maxpool7(x)

        # x = self.conv8(x)
        # x = F.relu(x)
        # x = self.batchnorm8(x)
        # x = self.maxpool8(x)
        #
        # x = self.conv9(x)
        # x = F.relu(x)
        # x = self.batchnorm9(x)
        # x = self.maxpool9(x)

        x = self.flatten(x)

        x = self.linear1(x)
        embedding = F.relu(x)

        output = self.linear2(embedding)

        return embedding, output


if __name__ == "__main__":
    auto_encoder = CVCNN()

    input = torch.randn((10, 1,2, 128))

    output = auto_encoder(input)

    print(output[0].shape)
    print(output[1].shape)
    def count_parameters_in_MB(model):
        return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    num_params = count_parameters_in_MB(auto_encoder)
    print(f'Number of parameters: {num_params}')