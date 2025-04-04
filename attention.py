import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embedded: int, input_projection_bias=True, output_projection_bias=True):
        super().__init__()

        self.input_proj = nn.Linear(d_embedded, 3 * d_embedded, bias = input_projection_bias)
        self.output_proj = nn.Linear(d_embedded, d_embedded, bias = output_projection_bias)
        self.n_heads = n_heads
        self.d_head = d_embedded // n_heads

    def forward(self, x: torch.Tensor, casual_mask = False):
        #x -> (Batch_Size, Sequence_Length, Dimension)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Sequence_Length, Dimesion) -> (Batch_Size, Sequence_Length, Dimension * 3), into 3 tensor shape
        q, k , v = self.input_proj(x).chunk(3, dim = -1)

        # (Batch_Size, Sequence_Length, Dimension) -> (Batch_Size, Sequence_Length, Dimension / Heads) -> (Batch_Size, H, SeqLength, Dim / H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (Batch_Size, H, Sequence_Length, Dimension / Head)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            mask = torch.ones_likes(weight, dtype = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        # (Batch_Size, H, SeqLength, SeqLength) @ (Batch_Size, H, SeqLen, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.output_proj(output)

        # (Batch_Size, Sequence_Length, Dimension)
        return output
