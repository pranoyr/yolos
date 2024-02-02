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


class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[-1] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)

	def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'pred_logits' in outputs
		src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
									dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
		losses = {'loss_ce': loss_ce}

		if log:
			# TODO this should probably be a separate loss, not hacked in this one here
			losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
		return losses

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, indices, num_boxes):
		""" Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
		This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		"""
		pred_logits = outputs['pred_logits']
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
		# Count the number of predictions that are NOT "no-object" (which is the last class)
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {'cardinality_error': card_err}
		return losses

	def loss_boxes(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		   targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		   The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['pred_boxes'][idx]
		target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses['loss_bbox'] = loss_bbox.sum() / num_boxes

		loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
			box_ops.box_cxcywh_to_xyxy(src_boxes),
			box_ops.box_cxcywh_to_xyxy(target_boxes)))
		losses['loss_giou'] = loss_giou.sum() / num_boxes
		return losses

	def loss_masks(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the masks: the focal loss and the dice loss.
		   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		"""
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)
		src_masks = outputs["pred_masks"]
		src_masks = src_masks[src_idx]
		masks = [t["masks"] for t in targets]
		# TODO use valid to mask invalid areas due to padding in loss
		target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
		target_masks = target_masks.to(src_masks)
		target_masks = target_masks[tgt_idx]

		# upsample predictions to the target size
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
								mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks.flatten(1)
		target_masks = target_masks.view(src_masks.shape)
		losses = {
			"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
			"loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
		return losses

	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'cardinality': self.loss_cardinality,
			'boxes': self.loss_boxes,
			'masks': self.loss_masks
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets)

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		# if is_dist_avail_and_initialized():
		#     torch.distributed.all_reduce(num_boxes)
		# num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

		# In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
		if 'aux_outputs' in outputs:
			for i, aux_outputs in enumerate(outputs['aux_outputs']):
				indices = self.matcher(aux_outputs, targets)
				for loss in self.losses:
					# if loss == 'masks':
					#     # Intermediate masks losses are too costly to compute, we ignore them.
					#     continue
					kwargs = {}
					if loss == 'labels':
						# Logging is enabled only for the last layer
						kwargs = {'log': False}
					l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
					l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
					losses.update(l_dict)

		return losses



class HungarianMatcher(nn.Module):
	"""This class computes an assignment between the targets and the predictions of the network

	For efficiency reasons, the targets don't include the no_object. Because of this, in general,
	there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
	while the others are un-matched (and thus treated as non-objects).
	"""

	def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
		"""Creates the matcher

		Params:
			cost_class: This is the relative weight of the classification error in the matching cost
			cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
			cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
		"""
		super().__init__()
		self.cost_class = cost_class
		self.cost_bbox = cost_bbox
		self.cost_giou = cost_giou
		assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

	@torch.no_grad()
	def forward(self, outputs, targets):
		""" Performs the matching

		Params:
			outputs: This is a dict that contains at least these entries:
				 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
				 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

			targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
				 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
						   objects in the target) containing the class labels
				 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

		Returns:
			A list of size batch_size, containing tuples of (index_i, index_j) where:
				- index_i is the indices of the selected predictions (in order)
				- index_j is the indices of the corresponding selected targets (in order)
			For each batch element, it holds:
				len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
		"""
		bs, num_queries = outputs["pred_logits"].shape[:2]

		# We flatten to compute the cost matrices in a batch
		out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
		out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

		# print(out_bbox)

		# Also concat the target labels and boxes
		tgt_ids = torch.cat([v["labels"] for v in targets])
		tgt_bbox = torch.cat([v["boxes"] for v in targets])

		# Compute the classification cost. Contrary to the loss, we don't use the NLL,
		# but approximate it in 1 - proba[target class].
		# The 1 is a constant that doesn't change the matching, it can be ommitted.
		cost_class = -out_prob[:, tgt_ids]

		# Compute the L1 cost between boxes
		cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

		# Compute the giou cost betwen boxes
		cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

		# Final cost matrix
		C = self.cost_bbox * cost_bbox + self.cost_class * cost_class  + self.cost_giou * cost_giou
		C = C.view(bs, num_queries, -1).cpu()

		#replace nan with 1
		C[C != C] = 1

		sizes = [len(v["boxes"]) for v in targets]
		indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
		return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
	return HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)


def exists(val):
	return val is not None


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


		# self.class_token = nn.Parameter(torch.randn(1, 1, dim))
		self.detn_tokens = nn.Parameter(torch.randn(1, max_dets, dim))
			
	
		num_patches = (image_size // patch_size) ** 2  
		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 extra for class token
		self.pos_enc_dets =  nn.Parameter(torch.randn(1, max_dets, dim)) # 1 extra for class token
  
		checkpoint = torch.load('outputs/yolo/checkpoints/vit.pt')
		with torch.no_grad():
			self.pos_enc.copy_(checkpoint['state_dict']['pos_enc'])



		self.pos_enc1 = nn.Parameter(torch.cat([self.pos_enc[:,1:,:], self.pos_enc_dets], dim=1))
  
	
		
		self.encoder = Encoder(dim, n_heads, d_head, depth)
  
		self.norm = nn.LayerNorm(dim)
		
  

		self.box_embeds = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, 4),
			nn.ReLU()
		)

		self.class_embeds = nn.Linear(dim, num_classes+1)
		
	def forward(self, x):
		# (batch_size, channels, height, width) --> (batch_size, timesteps, features)
		x = self.to_patch_embedding(x)
	
		
		# add class token
		detn_tokens = repeat(self.detn_tokens, '1 n d -> b n d', b=x.shape[0])
		# x, _ = pack([class_token, x], "b * d")
		x, _ = pack([x, detn_tokens], "b * d")

		# # sum of pose encodings
		# print(self.pos_enc[:, 1:, :].sum())
		# print(self.pos_enc1[:, :-100, :].sum())

		x += self.pos_enc1
		# transformer encoder
		x = self.encoder(x)
  
		x =  self.norm(x)

		# get the detection tokens output  , last 100 tokens
		x = x[:, -self.max_dets:, :]

    
  
		boxes = self.box_embeds(x)
		logits = self.class_embeds(x)

		return boxes, logits



class YoloS(torch.nn.Module):
	def __init__(self, dim=1024, image_size=512, patch_size=32, n_heads=2, d_head=64, depth=6, max_dets=2, num_classes=91):
		super(YoloS, self).__init__()

		self.model = ViT(dim=512, image_size=512, patch_size=16, n_heads=8, d_head=64, depth=6, max_dets=100, num_classes=91)

		checkpoint = torch.load('outputs/yolo/checkpoints/vit.pt')
		self.model.load_state_dict(checkpoint['state_dict'], strict=False)


		matcher = build_matcher()
		weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}

		losses = ['labels', 'boxes', 'cardinality']
		self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
							eos_coef=0.1, losses=losses)
		
	def forward(self, imgs, targets=None):	
		boxes , logits = self.model(imgs)
		outputs = {"pred_logits": logits, "pred_boxes": boxes}
	
		loss = None
		if exists(targets):
			loss = self.criterion(outputs, targets)

		return outputs, loss


if __name__=="__main__":
	# imgs
	imgs = torch.randn(2, 3, 512, 512)
	# target
	targets = [{"labels": torch.randint(0, 10, (5,)), "boxes": torch.randint(0, 10, (5, 4)).float()} for _ in range(2)]

	model = YoloS(dim=512, image_size=512, patch_size=32, n_heads=2, d_head=64, depth=6, max_dets=100, num_classes=91)
	model.eval()
 
	# inference
	detections, loss = model(imgs, targets)
	print(detections['pred_logits'].shape)
	print(detections['pred_boxes'].shape)

# model = ViT(dim=512, image_size=512, patch_size=16, n_heads=8, d_head=64, depth=6, max_dets=100, num_classes=1000)
# checkpoint = torch.load('outputs/yolo/checkpoints/Yolo.pt')
# model.load_state_dict(checkpoint['state_dict'])



	