import torch.nn as nn
import torch
import torch.nn.functional as F

feat = torch.randn(4, 2, 5, 5)

# feat = feat.transpose(0, 1)


# attn_mask = torch.randn(1, 5, 4)
# attn_mask = attn_mask.sigmoid() > 0.2  # True/False matrix
# attn_mask = attn_mask.detach()
# print(attn_mask)
# print(attn_mask.sum(-1))
# print(attn_mask.shape[-1])
# attn_mask[torch.where(
#     attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # every h*w q
# print(attn_mask)
# print(feat.shape, feat)
# feat = feat.argmax(dim=1)
# print(feat.shape, feat)
#
#
# def save_feature_map(feat, save_path, prefix):
#     import matplotlib.pyplot as plt
#     import os
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     # feat = feat.flatten(0, 1)
#     feat = feat.permute(1, 2, 0)
#     for i in range(feat.shape[-1]):
#         plt.imshow(feat[:, :, i], cmap='gray')
#         plt.axis('off')
#         plt.savefig(os.path.join(save_path, prefix + "_{}.jpg".format(i)), bbox_inches='tight', pad_inches=0)
#     print(prefix, 'done')
#
# save_feature_map(feat, 'vis', 'train')
