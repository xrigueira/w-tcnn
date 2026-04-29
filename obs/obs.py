
"""
Code I used to define the CNN model but turned out to be a wrong approach
"""

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
