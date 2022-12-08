from mmseg.models.builder import BACKBONES
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, build_upsample_layer)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmseg.models.backbones import ResNet
import torch.nn as nn
from torch.nn import Conv2d as Conv2d

@BACKBONES.register_module()
class SiameseResNet(ResNet):
    def __init__(self, mid_channels=256, **kwargs):
        super(SiameseResNet, self).__init__(**kwargs)
        self.mid_channels = mid_channels
        self.conv1x1 = nn.ModuleList()
        for i in range(4):
            self.conv1x1.append(ConvModule(
                in_channels=64 * 2 ** i,
                out_channels=mid_channels,
                kernel_size=1,
            ))

    def forward(self, x1, x2, interact_layer=[1, 2, 3]):
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x

        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)

        res1_outs = []
        res2_outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            res1_outs.append(self.conv1x1[i](x1))
            res2_outs.append(self.conv1x1[i](x2))

        return res1_outs, res2_outs


@BACKBONES.register_module()
class testResNet(SiameseResNet):
    def __init__(self, mid_channels=256, **kwargs):
        super(testResNet, self).__init__(**kwargs)

    def forward(self, x1, x2, interact_layer=[1, 2, 3]):
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x

        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)

        res1_outs = []
        res2_outs = []
        res = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            res1_outs.append(self.conv1x1[i](x1))
            res2_outs.append(self.conv1x1[i](x2))
            res.append(res2_outs[i]-res1_outs[i])

        return res, res1_outs, res2_outs

@BACKBONES.register_module()
class PSResNet(SiameseResNet):
    def __init__(self, mid_channels=256, **kwargs):
        super(PSResNet, self).__init__(**kwargs)

    def forward(self, x1, x2, interact_layer=[1, 2, 3]):
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x

        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)

        res1_outs = []
        res2_outs = []
        res = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            res1_outs.append(self.conv1x1[i](x1))
            res2_outs.append(self.conv1x1[i](x2))
            res.append(res2_outs[i]-res1_outs[i])

        return res, res1_outs, res2_outs

@BACKBONES.register_module()
class SiameseResNet_FPN(SiameseResNet):
    def __init__(self, mid_channels=256, **kwargs):
        super(SiameseResNet_FPN, self).__init__(**kwargs)
