import torch.nn as nn
import torch
from einops import rearrange, repeat, pack



rand = torch.tensor([[[1, 2, 3, 4 ,5],
                        [6, 7, 8, 9, 10]]]).float()


print(rand.shape)     

detn_tokens = nn.Parameter(rand)

clas_tokens = repeat(detn_tokens, '1 n d -> b n d', b=3)

pos = nn.Parameter(torch.randn(1, 12, 5))

print(clas_tokens)


x = torch.tensor(torch.rand(3,10, 5))

y, _ = pack([clas_tokens, x], "b * d")

z1 = y + pos
z2 = y + pos[:, :12]

print(z1.shape)
print(z2.shape)

# check if z1 and z2 are the same
print(torch.all(torch.eq(z1, z2)))