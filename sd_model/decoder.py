import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class AttentionBlock(nn.Module):

    def __init__(self, channels, int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (Batch_Size, Features, Height, Width)

        residue = x;

        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Featuers, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        x += residue

        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(32, input_channels)
        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)

        self.groupnorm_2 = nn.GroupNorm(32, output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1)

        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, kernel_size = 1, padding = 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (Batch_Size, Input_Channels, Height, Width)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            
            #Reverse process of encoder
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),

            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),

            ResidualBlock(512, 512),

            AttentionBlock(512),

            ResidualBlock(512, 512),

            ResidualBlock(512, 512),

            ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512),

            #Becomes / 4
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            #Becomes / 2
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256), 

            nn.Upsample(scale_factor = 2),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),

            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.GroupNorm(32, 128), 
            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size = 3, padding = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x /= 0.18215
        for module in self:
            x = module(x)

        return x