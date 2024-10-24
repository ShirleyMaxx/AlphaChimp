import inspect
import warnings
from abc import ABCMeta, abstractmethod

import math
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)

from typing import Dict, Tuple



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model / n_heads must be an integer"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.attn_fac = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask, attn_only=False):
        bs, seq_len = query.size()[:2]
        q = self.wq(query).reshape(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(bs * self.n_heads, seq_len, self.head_dim)
        k = self.wk(key).reshape(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(bs * self.n_heads, seq_len, self.head_dim)
        v = self.wv(value).reshape(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(bs * self.n_heads, seq_len, self.head_dim)
        # Now we get each (bs, n_heads, seq_len, head_dim).
        # Now after matmul we get seq_len * seq_len mat.
        scores = torch.bmm(q, k.transpose(-2, -1)) / self.attn_fac
        scores = scores.reshape(bs, self.n_heads, seq_len, seq_len)
        if mask is not None:
            # Notice we need broadcast mask.
            mask = mask.reshape(bs, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)


        if attn_only:
            # TODO: make this more reasonable.
            # We sum the scores from each head to get a total score.
            return scores.sum(dim=1)
        attention = nn.functional.softmax(scores, dim=-1).reshape(bs * self.n_heads, seq_len, seq_len)

        # print(attention.size())
        # print(value.size())
        out = torch.bmm(attention, v).reshape(bs, self.n_heads, seq_len, self.head_dim).transpose(1, 2)
        out = out.reshape(bs, seq_len, self.d_model)
        out = self.fc_out(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

    def get_forward_attention(self, src, src_mask):
        return self.self_attn(src, src, src, src_mask, attn_only=True)


@MODELS.register_module()
class ChimpTrackingHead(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 num_layers: int = 3,
                 embed_dim: int = 512,
                 num_heads: int = 16,
                 dim_ffn: int = 2048,
                 dropout: float = 0.1,
                 loss_score_cfg: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        super(ChimpTrackingHead, self).__init__(data_preprocessor=data_preprocessor)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.loss_score = MODELS.build(loss_score_cfg)
        
        self.transformer_enc = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_enc.append(TransformerEncoderLayer(embed_dim, num_heads, dim_ffn, dropout))


    def _forward_loss(self, assigned_feats: list, assigned_feats_pair: list, assigned_attn_mask: list):
        num_batch = len(assigned_feats)
        embed_dim = assigned_feats[0].size(1)
        assert embed_dim == self.embed_dim, 'Dims must match.'

        # TODO modify this to parallel.
        attn_results = []
        for i, (feat, feat_pair, attn_mask) in enumerate(zip(assigned_feats, assigned_feats_pair, assigned_attn_mask)):
            feat_len, feat_pair_len = feat.size(0), feat_pair.size(0)

            enc_input = torch.cat([feat, feat_pair], dim=0).unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0)
            # print(enc_input.size(), attn_mask.size())
            for n in range(self.num_layers - 1):
                enc_input = self.transformer_enc[n](enc_input, attn_mask)
            attn_scores = self.transformer_enc[-1].get_forward_attention(enc_input, attn_mask)
            attn_scores = attn_scores.squeeze(0)
            attn_scores = attn_scores[:feat_len, feat_len:]
            attn_results.append(attn_scores)
        # print('Batch ++')

        return attn_results

    def _forward_predict(self, predict_feats: list):
        num_batch = len(predict_feats)

        # TODO modify this to parallel.
        attn_results = []
        for batch_idx in range(num_batch):
            feat = predict_feats[batch_idx]['feat']
            feat_pair = predict_feats[batch_idx]['feat_pair']
            feat_len, feat_pair_len = feat.size(0), feat_pair.size(0)

            attn_mask = torch.cat(
                [feat.new_zeros((feat_len, )), feat.new_full((feat_pair_len, ), 1.0, dtype=torch.float32)],
                dim=0).unsqueeze(0).repeat(feat_len, 1)
            attn_mask_pair = torch.cat(
                [feat.new_full((feat_len, ), 1.0, dtype=torch.float32), feat.new_zeros((feat_pair_len, ))],
                dim=0).unsqueeze(0).repeat(feat_pair_len, 1)
            attn_mask = torch.cat([attn_mask, attn_mask_pair], dim=0).unsqueeze(0)

            enc_input = torch.cat([feat, feat_pair], dim=0).unsqueeze(0)
            for n in range(self.num_layers - 1):
                enc_input = self.transformer_enc[n](enc_input, attn_mask)
            attn_scores = self.transformer_enc[-1].get_forward_attention(enc_input, attn_mask)
            attn_scores = attn_scores.squeeze(0)
            attn_scores = attn_scores[:feat_len, feat_len:]
            attn_results.append(attn_scores)

        return attn_results

    def loss(self, assigned_feats: list, assigned_feats_pair: list, assigned_attn_mask: list, data_samples: list, **kargs):
        scores = self._forward_loss(assigned_feats, assigned_feats_pair, assigned_attn_mask)
        score_loss = self.loss_score(scores, data_samples)

        losses = {'loss_score': score_loss}
        return losses

    def predict(self, predict_feats: list, data_samples: list, **kargs):
       attn_results = self._forward_predict(predict_feats)
       for i in range(len(predict_feats)):
           predict_feats[i]['attn_scores'] = attn_results[i]
       return predict_feats