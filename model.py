import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Building Blocks
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class TemporalModule(nn.Module):
    def __init__(self, channels, kernel_size=3, num_layers=3, dilation_base=2):
        super(TemporalModule, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (0, dilation * (kernel_size - 1) // 2)
            layers.append(nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                    dilation=(1, dilation), padding=padding))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.conv(x)

# ------------------------------
# Generator Model
# ------------------------------
class ResidualComplexAutoencoderTemporal(nn.Module):
    def __init__(self):
        super(ResidualComplexAutoencoderTemporal, self).__init__()
        self.enc_conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.enc_relu1 = nn.ReLU()
        self.res1 = ResidualBlock(64)
        
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_relu2 = nn.ReLU()
        self.res2 = ResidualBlock(128)
        
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_relu3 = nn.ReLU()
        self.res3 = ResidualBlock(256)
        
        self.temporal = TemporalModule(256, kernel_size=3, num_layers=3, dilation_base=2)
        
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, mag, phase):
        x = torch.cat((mag, phase), dim=1)
        
        x = self.enc_relu1(self.enc_conv1(x))
        x = self.res1(x)
        x = self.enc_relu2(self.enc_conv2(x))
        x = self.res2(x)
        x = self.enc_relu3(self.enc_conv3(x))
        x = self.res3(x)
        
        x = self.temporal(x)
        
        mask = self.mask_decoder(x)
        mask_complex = torch.complex(mask[:, 0, :, :], mask[:, 1, :, :])
        
        noisy_mag_linear = torch.expm1(mag.squeeze(1))
        noisy_phase = phase.squeeze(1)
        noisy_complex = torch.polar(noisy_mag_linear, noisy_phase)
        
        target_freq = noisy_complex.size(1)
        target_time = noisy_complex.size(2)
        mask_complex = mask_complex[:, :target_freq, :target_time]
        
        enhanced_complex = mask_complex * noisy_complex + noisy_complex
        return enhanced_complex

# ------------------------------
# Discriminator Model
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            x = torch.randn(1, 2, input_shape[0], input_shape[1])
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flattened_size = x.numel() // x.size(0)
        
        self.linear = nn.Linear(self.flattened_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        return self.sigmoid(x)
