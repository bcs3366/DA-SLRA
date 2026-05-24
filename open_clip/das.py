import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from utils import read_json
from functools import partial
from .swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from .vit import VisionTransformer, interpolate_pos_embed
from .bert import BertModel, BertConfig
from .resnet import resnet50, resnet101
from torchvision.models import vgg16, vgg19_bn
from torchvision import models
from torch.autograd import Variable
from typing import Tuple
from torch import nn, Tensor

def compute_u_from_m(margin, device, dtype,
                     m_min=0.0, m_max=0.5, eps=1e-12,
                     mode="cos", k=5.0, x0=0.5):
    m = float(abs(margin))
    m = max(m_min, min(m, m_max))
    denom = max(m_max - m_min, eps)
    x = (m - m_min) / denom
    x_t = torch.tensor(x, device=device, dtype=dtype)
    if mode == "pow2":
        u = (1.0 - x_t) ** 2
    elif mode == "sqrt":
        u = 1.0 - torch.sqrt(torch.clamp(x_t, min=0.0))
    elif mode == "cos":
        u = torch.cos(0.5 * math.pi * x_t)
    elif mode == "sigmoid":
        u = torch.sigmoid(k * (x0 - x_t))
    else:
        ekx = torch.exp(-k * x_t)
        ek  = math.exp(-k)
        u = (ekx - ek) / (1.0 - ek)


    u = torch.clamp(u, 0.0, 1.0)
    return u


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    d = x.size(0)
    assert x.dim() == 2 and x.size(0) == x.size(1)
    mask = ~torch.eye(d, device=x.device, dtype=torch.bool)
    return x.masked_select(mask)


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply

def build_vision_encoder(config, load_vision_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2
    if config['use_swin']:
        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']


        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)
        if load_vision_params:
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']


            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

        assert config['patch_size'] == 16
        vision_width = 384

        vision_encoder = VisionTransformer(
            img_size=config['image_res'],
            patch_size=config['patch_size'],
            embed_dim=384, depth=12, num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            local_attn_depth=4)

        if load_vision_params:
            state_dict = torch.load("data/deit_small_patch16_224-cd65a155.pth", map_location="cpu")["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches,
                                                       num_extra_tokens=1)
            state_dict['pos_embed'] = pos_embed_reshaped

    if load_vision_params:
        if config['use_swin']:
            print("### Load Trans-Encoder[SWin-T]: ", flush=True)
        else:
            print("### Load Trans-Encoder[ViT]: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict,
                                             strict=False)


    return vision_encoder, vision_width



def build_conv_encoder(config, load_vision_params=False, ins='resnet'):

    resnet_ckpt = config['resnet_ckpt']
    finetune_conv = config['finetune_conv']
    if ins == 'resnet':
        resnet_with_last = nn.Sequential(*list(resnet50(num_classes=30).children())[:-1])
        conv_width = 2048
    elif ins == 'vgg':
        vgg = vgg19_bn(num_classes=30)
        vgg.classifier[6] = nn.Linear(4096, 2048)
        resnet_with_last = vgg
        conv_width = 2048
    else:
        raise ValueError

    if load_vision_params:
        print("### Load Conv-Encoder[ResNet-50]: ", flush=True)
        state_dict = torch.load(resnet_ckpt, map_location="cpu")
        if len(state_dict) < 10:
            state_dict = state_dict['model']
        if ins == 'vgg':
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        resnet_with_last.load_state_dict(state_dict,
                                         strict=False)

        for child in resnet_with_last.children():
            for param in child.parameters():
                param.requires_grad = finetune_conv
    return resnet_with_last, conv_width


def build_text_encoder(config, load_text_params=False):
    text_config = read_json(config['text_config'])
    text_width = text_config['hidden_size']
    bert_config = BertConfig.from_json_file(config['text_config'])
    text_encoder = BertModel(bert_config)


    if load_text_params:

        print("### Load Trans-Encoder[Bert-B]: ", flush=True)
        init_checkpoint = config['text_encoder'] + '/pytorch_model.bin'
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        text_encoder.load_state_dict(state_dict, strict=False)

        for child in text_encoder.children():
            for param in child.parameters():
                param.requires_grad = True

    return text_encoder, text_width


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim))


def clones(module, N):
    """Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_pretrained_das(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2
    print("### Loading pretrained vision encoder", flush=True)


    window_size = read_json(config['vision_config'])['window_size']

    for k in list(state_dict.keys()):
        if 'relative_position_bias_table' in k:
            dst_num_pos = (2 * window_size - 1) ** 2
            state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
        elif ('relative_position_index' in k) or ('attn_mask' in k):
            del state_dict[k]

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

    return state_dict


def convert_label_to_similarity(normed_feature: Tensor,
                                label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class rec_Loss(nn.Module):
    def __init__(self, m: float = 0.25, gamma: float = 64.0) -> None:
        super(rec_Loss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(-sp.detach() + 1.0 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1.0 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n =  an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        )
        return loss

class DAS_baseline(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=True,
                 use_contrastive_loss=False, use_affil_loss=False):
        super().__init__()
        if config['is_das']:
            self.embed_dim = config['embed_dim']
            self.temp = nn.Parameter(torch.ones([]) * config['temp1'])

    def encode_image(self, visual_encoder, vision_proj, detach_tokens=False):
        raise NotImplementedError

    def encode_text(self, text_encoder, text_proj, detach_tokens=False):
        raise NotImplementedError

    def load_pretrained_das(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained_das(ckpt_rpath, config, is_eval=is_eval, load_text=True)  # ckpt_rpath 是权重文件的路径
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image):
        """
        vision_embeds: cls + patch embeds
        """
        return F.normalize(self.vision_proj(self.vision_encoder(image))[:, 0, :])

    def get_text_embeds(self, text_ids):
        """
        text_embeds: cls + sequence embeds
        """
        return F.normalize(self.text_proj(self.text_encoder(text_ids))[:, 0, :])

    def get_contr_loss(self, image_feat, text_feat, idx=None, label=None, config=None):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim


        logits = image_feat @ text_feat.t() / self.temp  # [batch_size, batch_size]
        bsz = image_feat.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0), "idx 与 batch 大小不匹配"
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)


            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2


    def get_affil_loss(self, image_feat, text_feat, idx=None, label=None, config=None):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        # logits = image_feat @ text_feat.t()
        la_idx = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=1).t()).float()
        # calculate centers of clusters
        img_centers = []
        txt_centers = []
        for i in range(image_feat.shape[0]):
            # # calculate mean of each cluster
            mod = la_idx[i].unsqueeze(dim=1)
            mask = mod.repeat(1, 512)
            non_zero_num = torch.sum(mod, dim=0)
            img_center = (image_feat * mask).sum(dim=0, keepdim=True) / non_zero_num
            txt_center = (text_feat * mask).sum(dim=0, keepdim=True) / non_zero_num

            img_centers.append(img_center)
            txt_centers.append(txt_center)

        img_centers = torch.cat(img_centers, dim=0)
        txt_centers = torch.cat(txt_centers, dim=0)

        img_centers_all = allgather(img_centers, torch.distributed.get_rank(), torch.distributed.get_world_size())
        txt_centers_all = allgather(txt_centers, torch.distributed.get_rank(), torch.distributed.get_world_size())

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())

        img2txt_center = image_feat_all @ txt_centers_all.t() / self.temp2
        txt2img_center = text_feat_all @ img_centers_all.t() / self.temp2

        bsz = img2txt_center.shape[0]
        labels = torch.eye(bsz, device=image_feat.device)

        loss_i2t = -torch.sum(F.log_softmax(img2txt_center, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(txt2img_center.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_triplet_loss(self, image_feat, text_feat, margin=0.2, max_violation=False):


        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        scores = image_feat_all @ text_feat_all.t()  # 相似度矩阵


        bsz = image_feat_all.shape[0]
        diagonal = scores.diag().view(bsz, 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda(device=image_feat.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        sum_cost_s = cost_s.sum()
        sum_cost_im = cost_im.sum()
        return sum_cost_s + sum_cost_im

    def weighted_triplet_loss(self, image_feat, text_feat, margin=0.2, gamma=1.0, max_violation=False):
        scores = image_feat @ text_feat.t()
        bsz = image_feat.shape[0]
        diagonal = scores.diag().view(bsz, 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)

        mask = torch.eye(bsz, dtype=torch.bool, device=image_feat.device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        p_s = torch.exp(-cost_s)
        weights_s = (1 - p_s) ** gamma
        p_im = torch.exp(-cost_im)
        weights_im = (1 - p_im) ** gamma
        cost_s = weights_s * cost_s
        cost_im = weights_im * cost_im
        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return (cost_s.sum() + cost_im.sum()) / 2.0









