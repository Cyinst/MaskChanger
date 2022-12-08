# Copyright (c) Open-CD. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule, Conv2d, build_activation_layer, build_norm_layer, build_plugin_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, Sequential, ModuleList
from ..losses import accuracy

import matplotlib.pyplot as plt
import os

def save_feature_map(feat, save_path='vis', prefix='test'):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if prefix != 'gt':
        feat = feat.argmax(dim=1)
    else:
        feat = feat.flatten(0, 1)
    feat = feat.permute(1, 2, 0)
    feat = feat.detach().cpu().numpy()
    for i in range(feat.shape[-1]):
        plt.imshow(feat[:, :, i], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, prefix + "_{}.jpg".format(i)), bbox_inches='tight', pad_inches=0)


@HEADS.register_module()
class Mask2FormerHead(BaseDecodeHead):
    def __init__(self,
                 feat_channels=256,  # 256
                 out_channels=256,  # 256
                 in_channels=None,
                 num_queries=100,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 interpolate_mode='bilinear',
                 dropout_ratio=0.1,
                 **kwargs):
        super().__init__(input_transform='multiple_select',
                         in_channels=[256, 256, 256, 256], **kwargs)
        if in_channels is None:
            in_channels = [256, 256, 256, 256]
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.num_classes = 2
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers. \
            attn_cfgs.num_heads  # 8
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers. \
                   attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=[256, 256, 256, 256],
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat2 = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            # nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.segconv = nn.Sequential(
            ConvModule(
                in_channels=self.num_queries,
                out_channels=self.num_queries,
                kernel_size=1,
            ),
            nn.Conv2d(self.num_queries, self.num_classes, kernel_size=1)
        )
        self.conv3x3 = nn.Sequential(
            ConvModule(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=3,
                padding=1),

        )

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)  # shape (batch_size, num_queries, c)
        # cls_pred = self.cls_embed(decoder_out)  # linear, shape (batch_size, num_queries(100), num_classes+1(134))
        mask_embed = self.mask_embed(decoder_out)  # mlp, shape (batch_size, num_queries(100), out_channels(256))
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)  # downsample
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5  # True/False matrix
        attn_mask = attn_mask.detach()
        mask_pred = self.segconv(mask_pred)
        return mask_pred, attn_mask  # (b,q,h_max,w_max), (b*(nh 8),q,h*w)

    def siamese_forward(self, decoder_out1, decoder_out2, mask_feature, attn_mask_target_size):
        mask_pred1, attn_mask1 = self.forward_head(
            decoder_out1, mask_feature, attn_mask_target_size)
        mask_pred2, attn_mask2 = self.forward_head(
            decoder_out2, mask_feature, attn_mask_target_size)
        attn_mask = attn_mask1 + attn_mask2
        attn_mask = attn_mask.sigmoid() < 0.5  # True/False matrix
        attn_mask = attn_mask.detach()
        mask_pred = mask_pred1 + mask_pred2
        return mask_pred, attn_mask  # (b,q,h_max,w_max), (b*(nh 8),q,h*w)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        feat, feats1, feats2 = feats

        mask_features1, multi_scale_memorys1 = self.pixel_decoder(feats1)  # (b,c,h_max,w_max), [bchw](low to high)
        mask_features2, multi_scale_memorys2 = self.pixel_decoder(feats2)  # (b,c,h_max,w_max), [bchw](low to high)
        # multi_scale_memorys1 = []
        # multi_scale_memorys2 = []
        # for i in range(len(feats2)):
        #     multi_scale_memorys2.append(feats2[3 - i])
        # mask_features = feat[0] + resize(
        #     input=feat[1],
        #     size=feat[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=False)
        # mask_features = self.conv3x3(mask_features)
        mask_features = mask_features2-mask_features1
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs2 = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):  # 3
            decoder_input2 = self.decoder_input_projs[i](multi_scale_memorys2[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input2 = decoder_input2.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)  # (1,1,c)
            decoder_input2 = decoder_input2 + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input2.new_zeros(
                (batch_size,) + multi_scale_memorys2[i].shape[-2:],
                dtype=torch.bool)  # (batch_size, h, w)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs2.append(decoder_input2)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat2 = self.query_feat2.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        mask_pred_list = []
        # (b,q 100,nc 134), (b,q,h_max,w_max), (b*(nh 8),q,h_min*w_min)
        mask_pred, attn_mask = self.forward_head(
            query_feat2, mask_features, multi_scale_memorys2[0].shape[-2:])
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):  # 9
            level_idx = i % self.num_transformer_feat_level  # 3
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # every h*w q

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat2 = layer(
                query=query_feat2,
                key=decoder_inputs2[level_idx],
                value=decoder_inputs2[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            mask_pred, attn_mask = self.forward_head(
                query_feat2, mask_features, multi_scale_memorys2[
                                                             (i + 1) % self.num_transformer_feat_level].shape[-2:])

            mask_pred_list.append(mask_pred)
        return mask_pred_list

    def mask_loss(self, seg_logits_list, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = resize(
                input=seg_logits_list[i],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=False)
        seg_label_ = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            for i in range(len(seg_logits_list)):
                if self.sampler is not None:
                    seg_weight = self.sampler.sample(seg_logits_list[i], seg_label)
                else:
                    seg_weight = None
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits_list[i],
                        seg_label_,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    loss['acc_seg'] = accuracy(
                        seg_logits_list[i], seg_label_, ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits_list[i],
                        seg_label_,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    loss['acc_seg'] += accuracy(
                        seg_logits_list[i], seg_label_, ignore_index=self.ignore_index)
            loss['acc_seg'] = loss['acc_seg'] // len(seg_logits_list)
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, img_metas)
        losses = self.mask_loss(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs, img_metas)
        # save_feature_map(torch.cat(seg_logits, dim=0), 'vis/' + img_metas[0]['filename'][0][-7:-4])
        return seg_logits[-1]
