import torch
from torch import nn
from torch.nn import functional as F

from decoder import AttentionBlock, ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128,128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128,128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2) 
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),

            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(128, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),

            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(256, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, Height / 8, Width / 8)
            AttentionBlock(512), 

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512),

            #Normalization
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 

            #Sigmoid Linear Unit 
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(512, 8, kernel_size = 3, padding = 1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #x: (Batch_Size, Channel, Height, Width)
        #noise: (Batch_Size, Output_Channels, Height / 8, Width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                #Left, Right, Top, Bottom
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

        #(Batch_Size, 8, Height / 8, Width / 8) -> Two Tensors of (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)

        log_variance = torch.clamp(log_variance, -30, 20)

        #Logarithmic to Exponential
        #(Batch_Size, 4, Height / 8, Width / 8)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance) = X?
        x = mean + stdev * noise

        #Scaling constant based on the research papers
        x *= 0.18215

        return x