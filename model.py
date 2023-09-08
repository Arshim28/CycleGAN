import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T

n_residual_blocks = 6

#ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.block(x)
        return out + x

#DownSampling
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):
        super(DownSampling, self).__init__()
        self.norm = norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instanceNorm2d = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.instanceNorm2d(x)
        
        output = self.relu(x)
        return output

#UpSampling
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSampling, self).__init__()
        self.Up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x):
        return self.Up(x)

#Generator
class Generator(nn.Module):
    def __init__(self, in_channels, n_residual_blocks):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.n_residual_blocks = n_residual_blocks
        
        #C7 S1-64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)

        )
        
        #d128
        self.down_block1 = DownSampling(64, 128)
        
        #d256
        self.down_block2 = DownSampling(128, 256)
        
        #R256 x 6
        self.residual_block = ResidualBlock(256)
        
        #U128
        self.up_block1 = UpSampling(256, 128)
        
        #U64
        self.up_block2 = UpSampling(128, 64)
        
        #C7 s1-3
        self.conv2 = nn.Conv2d(64, in_channels, kernel_size=7, padding=3, padding_mode='reflect')
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.down_block1(x)
        x = self.down_block2(x)
        
        for _ in range(n_residual_blocks):
            x = self.residual_block(x)
        
        x = self.up_block1(x)
        x = self.up_block2(x)
        
        x = self.conv2(x)
        output = self.tanh(x)
        
        return output

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width  = input_shape
        self.output_shape = (1, height//2**4, width//2**4)
        self.layer = nn.Sequential(
            DownSampling(channels, 64, norm = False),
            DownSampling(64, 128),
            DownSampling(128, 256),
            DownSampling(256, 512),
            
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
        
        
    def forward(self, x):
        return self.layer(x)