"""I have to build the UNet model and train it on the dataset.
The dataloader maybe be the same as for the transformer model, 
with the only difference being the tgt column, which should
be the generated label for each window.

Go over the code in ChatGPT, Gemini and this website: 
https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
for an idea of how to build the model.

The dimensions resulting matrix after the convolution operation 
can be calculated with the following formula [(W - K + 2P) / S] + 1.

W is the input volume - in your case 128
K is the Kernel size - in your case 5
P is the padding - in your case 0 i believe
S is the stride - which you have not provided.

Why UNet?
It gives better results from what I have read:
https://arxiv.org/pdf/2002.09545.pdf
https://arxiv.org/pdf/1905.13628.pdf
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class DoubleConv(nn.Module):
    """Double Convolution Block for UNet model.
    This block consists of two convolutional layers followed by batch normalization
    and ReLU activation function.
    ----------
    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, n_variables, window_size, n_classes, input_channels, channels, d_fc) -> None:
        super(UNet, self).__init__()

        """UNet model for image segmentation.
        This model consists of an encoder and a decoder.
        ----------
        Args:
            - n_variables (int): Number of input variables.
            - channels (int): Number of channels.
            - n_class (int): Number of classes.
        
        Returns:
            - x (torch.Tensor): Output tensor.
        """

        # Define the encoder
        self.encoder1 = DoubleConv(input_channels, channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.encoder2 = DoubleConv(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.encoder3 = DoubleConv(channels * 2, channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.encoder4 = DoubleConv(channels * 4, channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.encoder5 = DoubleConv(channels * 8, channels * 16)

        # Define the decoder
        self.conv_transpose1 = nn.ConvTranspose2d(channels * 16, channels * 8, kernel_size=2, stride=1)
        self.decoder1 = DoubleConv(channels * 16, channels * 8)

        self.conv_transpose2 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=1)
        self.decoder2 = DoubleConv(channels * 8, channels * 4)

        self.conv_transpose3 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=1)
        self.decoder3 = DoubleConv(channels * 4, channels * 2)

        self.conv_transpose4 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=1)
        self.decoder4 = DoubleConv(channels * 2, channels)
        
        self.outconv = nn.Conv2d(channels, n_classes, kernel_size=1)
        
        # Define the linear layers
        self.fc1 = nn.Linear(input_channels * window_size * n_variables, d_fc)
        self.fc2 = nn.Linear(d_fc, d_fc // 2)
        self.fc3 = nn.Linear(d_fc // 2, d_fc // 4)
        self.fc4 = nn.Linear(d_fc // 4, n_classes)

    def forward(self, src):

        # Encoder
        xencoder1 = self.encoder1(src)
        xpool1 = self.pool1(xencoder1)

        xencoder2 = self.encoder2(xpool1)
        xpool2 = self.pool2(xencoder2)

        xencoder3 = self.encoder3(xpool2)
        xpool3 = self.pool3(xencoder3)

        xencoder4 = self.encoder4(xpool3)
        xpool4 = self.pool4(xencoder4)

        xencoder5 = self.encoder5(xpool4)

        # Decoder
        xconv_transpose1 = self.conv_transpose1(xencoder5)
        xcat1 = torch.cat((xconv_transpose1, xencoder4), dim=1)
        xdecoder1 = self.decoder1(xcat1)

        xconv_transpose2 = self.conv_transpose2(xdecoder1)
        xcat2 = torch.cat((xconv_transpose2, xencoder3), dim=1)
        xdecoder2 = self.decoder2(xcat2)

        xconv_transpose3 = self.conv_transpose3(xdecoder2)
        xcat3 = torch.cat((xconv_transpose3, xencoder2), dim=1)
        xdecoder3 = self.decoder3(xcat3)

        xconv_transpose4 = self.conv_transpose4(xdecoder3)
        xcat4 = torch.cat((xconv_transpose4, xencoder1), dim=1)
        xdecoder4 = self.decoder4(xcat4)
        
        xoutconv = self.outconv(xdecoder4)

        # Flatten the output tensor
        xoutconv = xoutconv.view(xoutconv.size(0), -1)

        # Linear layers
        x = relu(self.fc1(xoutconv))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        return x