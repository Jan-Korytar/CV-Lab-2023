import torch
import torch.nn as nn


class conv_block(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class unet(torch.nn.Module):
    def __init__(self, depth, base_dim, in_channels, out_channels):
        super().__init__()
        """ Encoder """
        self.depth = depth
        self.e1 = encoder_block(in_channels, base_dim)
        for i in range(2, depth + 1):
            exec(f'self.e{i} = encoder_block(int(base_dim), int(base_dim * 2))')
            base_dim *= 2


        """ Bottleneck """
        self.b = conv_block(base_dim, base_dim *2)
        base_dim *= 2
        """ Decoder """
        for i in range(1, depth + 1):
            exec(f'self.d{i} = decoder_block(int(base_dim), int(base_dim/2))')
            base_dim /= 2
        """ Classifier """
        self.outputs = nn.Conv2d(int(base_dim), out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        for i in range(2, self.depth + 1):
            exec(f's{i}, p{i} = self.e{i}(p{i-1})')

        """ Bottleneck """
        b = self.b(eval(f'p{i}'))
        """ Decoder """
        d1 = self.d1(b, eval(f's{i}'))
        for i in range(2, self.depth + 1):
            exec(f'd{i} = self.d{i}(d{i - 1}, s{self.depth + 1 - i})')
        """ Classifier """
        outputs = self.outputs(eval(f'd{i}'))
        return outputs
