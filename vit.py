import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
from softmax_attention import SoftmaxAttention
# from experiments.agent_attention import AgentAttention	
from switchhead_attention import SwitchHeadAttention
from torch.nn import functional as F
import time
from moe import MoELayer

from util import box_ops


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou



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
	def __init__(self, dim, hidden_dim=3072, dropout = 0.):
			super().__init__()
			self.net = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)



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

		self.self_attn = SoftmaxAttention(dim, n_heads, d_head, dropout=dropout)
		self.feed_forward = FeedForward(dim, hidden_dim=3072)
		#self.moe = MoELayer(input_dim=dim,  output_dim=dim, num_experts=6, sel_experts=2)
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
		#fc_out = self.moe(x_norm)

		# ADD
		x = fc_out + x
		return x



class ViT(nn.Module):
	def __init__(self, dim=512, image_size=512, patch_size = 16, n_heads = 8, d_head = 64, depth = 6, max_dets = 100, num_classes = 1000):
		super(ViT, self).__init__()
		
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


		self.class_token = nn.Parameter(torch.randn(1, 1, dim))
			
	
		num_patches = (image_size // patch_size) ** 2  
		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 extra for class token
		
		self.encoder = Encoder(dim, n_heads, d_head, depth)
  
		self.norm = nn.LayerNorm(dim)
		
  

		self.box_embed = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, 4),
			nn.ReLU()
		)

		self.class_embed = nn.Linear(dim, num_classes)
		# self.class_embed = nn.Lin
		
	def forward(self, x):
		# (batch_size, channels, height, width) --> (batch_size, timesteps, features)
		x = self.to_patch_embedding(x)
		# print(x.shape)
		
		# add class token
		class_token = repeat(self.class_token, '1 1 d -> b 1 d', b=x.shape[0])
		x, _ = pack([class_token, x], "b * d")

		# add positional encoding
		x += self.pos_enc
		# transformer encoder
		x = self.encoder(x)
  
		x =  self.norm(x)

		# get the detection tokens output  , last 100 tokens
		x = x[:, 0, :]

        
		x = self.class_embed(x)
  
		# boxes = self.box_embed(x)
		# logits = self.class_embed(x)

		return x


# x = torch.randn(2, 3, 512, 512)
# model = ViT(dim=512, image_size=512, patch_size=16, n_heads=8, d_head=64, depth=6, max_dets=100, num_classes=91)
# model.eval()
# x = model(x)
# print(x.shape)