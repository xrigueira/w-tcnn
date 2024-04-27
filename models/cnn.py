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

class Downscale(nn.Module):
    """[Used by _UNet] Downscale block for UNet model.
    This block consists of a DoubleConv block followed by a max pooling layer.
    ----------

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Upscale(nn.Module):
    """[Used by _UNet] Upscale block for UNet model.
    This block consists of an upsampling layer followed by a DoubleConv block.
    ----------
    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_encoder):
        x = self.conv_transpose(x)
        x = torch.cat([x, x_encoder], dim=1)
        x = self.double_conv(x)
        return x
    
class Encoder(nn.Module):
    """[Used by _UNet] Encoder block for UNet model.
    This block consists of four Downscale blocks.
    ----------
    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            Downscale(in_channels, out_channels),
            Downscale(out_channels, out_channels * 2),
            Downscale(out_channels * 2, out_channels * 4),
            Downscale(out_channels * 4, out_channels * 8),
            DoubleConv(out_channels * 8, out_channels * 16)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    """[Used by _UNet] Decoder block for UNet model.
    This block consists of four Upscale blocks.
    ----------
    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            Upscale(in_channels, out_channels),
            Upscale(out_channels, out_channels // 2),
            Upscale(out_channels // 2, out_channels // 4),
            Upscale(out_channels // 4, out_channels // 8)
        )

    def forward(self, x, x_encoder):
        x = self.decoder(x, x_encoder)
        return x

# Define the UNet model
class _UNet(nn.Module):
    """Non functional UNet model for anomaly detection. This model 
    consists of an encoder and a decoder. The problem is that it 
    does not pass the x_encoder to the decoder, so it can perform 
    the skip connections with concatenation.
    ----------
    Args:
        - n_class (int): Number of classes.
    
    Returns:
        - x (torch.Tensor): Output tensor.
    """

    def __init__(self, n_class) -> None:
        super(_UNet, self).__init__()

        self.encoder = Encoder(3, 64)
        self.decoder = Decoder(1024, 64)
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.outconv(x)
        return x

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, n_variables, channels, n_classes) -> None:
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
        self.encoder1 = DoubleConv(n_variables, channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(channels * 2, channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(channels * 4, channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = DoubleConv(channels * 8, channels * 16)

        # Define the decoder
        self.conv_transpose1 = nn.ConvTranspose2d(channels * 16, channels * 8, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(channels * 16, channels * 8)

        self.conv_transpose2 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(channels * 8, channels * 4)

        self.conv_transpose3 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(channels * 4, channels * 2)

        self.conv_transpose4 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(channels * 2, channels)
        
        self.outconv = nn.Conv2d(channels, n_classes, kernel_size=1)

    def forward(self, x):

        # Encoder
        xencoder1 = self.encoder1(x)
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

        return xoutconv