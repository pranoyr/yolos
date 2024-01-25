import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
from softmax_attention import SoftmaxAttention
from agent_attention import AgentAttention	
from switchhead_attention import SwitchHeadAttention
from torch.nn import functional as F
import time






class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		# we don't want to update this
		self.register_buffer("beta", torch.zeros(dim))

	def forward(self, x):
		return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
	"""https://arxiv.org/abs/2002.05202"""

	def forward(self, x):
		x, gate = x.chunk(2, dim=-1)
		return gate * F.gelu(x)


class FeedForward(nn.Module):
	def __init__(self, dim, mult=4):
		super().__init__()

		inner_dim = int(dim * mult * 2 / 3)
		self.ff = nn.Sequential(
			nn.Linear(dim, inner_dim * 2, bias=False),
			GEGLU(),
			LayerNorm(inner_dim),
			nn.Linear(inner_dim, dim, bias=False),
		)

	def forward(self, x):
		return self.ff(x)



class Encoder(nn.Module):
	def __init__(self, dim, n_heads, d_head, depth, dropout=0.):
		super().__init__()
	
		self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_head, dropout) for _ in range(depth)])
 
	def forward(self, x, context_mask=None):
		for layer in self.layers:
			x = layer(x, context_mask=context_mask)
		return x


class EncoderLayer(nn.Module):
	def __init__(self, dim, n_heads, d_head, dropout):
		super().__init__()

		self.self_attn = SwitchHeadAttention(dim, n_heads, d_head, dropout=dropout)
		self.feed_forward = FeedForward(dim)
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		
	def forward(self, x, context_mask=None):
		x_norm = self.norm1(x)
		# self attention
		attn_out = self.self_attn(x=x_norm, context_mask=context_mask)

		# ADD & NORM
		x = attn_out + x
		x_norm = self.norm2(x)

		# feed forward
		fc_out = self.feed_forward(x_norm)

		# ADD
		x = fc_out + x
		return x



class YoloS(nn.Module):
	def __init__(self, dim, image_size=256, patch_size = 64, n_heads = 8, d_head = 64, depth = 6, max_dets = 100):
		super(YoloS, self).__init__()
		
		self.dim = dim
		self.patch_size = patch_size
		self.max_dets = max_dets
		
		# number of features inside a patch
		self.patch_dim = patch_size * patch_size * 3
		
		self.to_patch_embedding = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
			nn.LayerNorm(self.patch_dim),
			nn.Linear(self.patch_dim, dim),
			nn.LayerNorm(dim))


		# self.class_token = nn.Parameter(torch.randn(dim))
			
		# 100 detection tokens	
		self.detn_tokens = nn.Parameter(torch.randn(max_dets, dim))
  
		num_patches = (image_size // patch_size) ** 2  
		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + max_dets, dim)) # 1 extra for class token

		self.encoder = Encoder(dim, n_heads, d_head, depth)
  

		self.mlp = nn.Linear(dim, 4)	
		


	def forward(self, x):
		# (batch_size, channels, height, width) --> (batch_size, timesteps, features)
		x = self.to_patch_embedding(x)
  

		# add class token
		class_token = repeat(self.detn_tokens, 'n d -> b n d', b=x.shape[0])
		x, _ = pack([class_token, x], "b * d")

		# add positional encoding
		x += self.pos_enc

		# transformer encoder
		x = self.encoder(x)

		# get the class token output
		# x = x[:, 0]
		# x = self.final_fc(x)

		# get the detection tokens output  , last 100 tokens
		dets = x[:, -self.max_dets:, :]
  
		dets = self.mlp(dets)

		return dets




model = YoloS(dim=512, image_size=512, patch_size=32, n_heads=2, d_head=64, depth=2, max_dets=100)


imgs = torch.randn(2, 3, 512, 512)
dets = torch.randn(2, 10, 5)  # batch_size, num_dets, 5


while True:
	start = time.time()
	dets =  model(imgs)
	print("FPS: ", 1.0 / (time.time() - start))

	