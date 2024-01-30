
import torch
from einops import rearrange, repeat

x = torch.rand(5, 20)  # (b, timesteps_q, dim)
y = torch.rand(5, 20)  # (b, timesteps_k, dim)

z = torch.stack([x, y], dim=0)  # (e , timesteps, dim)
print(z.shape)
print(z)


z = rearrange(z, 'e t d -> t e d')  # (timesteps, e, dim)
print(z.shape)
print(z)


import torch.nn as nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    

ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(512* 4, 512, bias=False)
        )
x = torch.randn(2, 10, 512)  # (b, timesteps_q, dim)
x = ff_out(x)
print(x.shape) # (b, timesteps, dim)

