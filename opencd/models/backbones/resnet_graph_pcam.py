# Copyright (c) Open-CD. All rights reserved.
from mmseg.models.builder import BACKBONES
from .resnet_pcam import *
from mmseg.models.backbones import ResNet
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from torch.nn import functional as F


class NLBlock1(BaseModule):
    def __init__(self, in_channels):
        super(NLBlock1, self).__init__()
        self.inter_channels = in_channels // 2
        self.g = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.W_z = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x = self.g(x1 - x2).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x2).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x1.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x1 - x2
        return z


class NLBlock2(BaseModule):
    def __init__(self, in_channels):
        super(NLBlock2, self).__init__()
        self.inter_channels = in_channels // 2
        self.g = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.g2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.W_z = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.W_z2 = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x1 = self.g(x1).view(batch_size, self.inter_channels, -1)
        g_x2 = self.g2(x2).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x2).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)
        y1 = torch.matmul(f_div_C, g_x1)
        y2 = torch.matmul(f_div_C, g_x2)
        y1 = y1.permute(0, 2, 1).contiguous()
        y2 = y2.permute(0, 2, 1).contiguous()
        y1 = y1.view(batch_size, self.inter_channels, *x1.size()[2:])
        y2 = y2.view(batch_size, self.inter_channels, *x1.size()[2:])

        W_y1 = self.W_z(y1)
        W_y2 = self.W_z2(y2)
        # residual connection
        z = W_y1 + x1 - x2 - W_y2
        return z

class NLBlock3(BaseModule):
    def __init__(self, in_channels):
        super(NLBlock3, self).__init__()
        self.inter_channels = in_channels // 2
        self.g = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.g2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta1 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi1 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.W_z = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.W_z2 = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x1 = self.g(x1).view(batch_size, self.inter_channels, -1)
        g_x2 = self.g2(x2).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x1 = self.theta1(x1).view(batch_size, self.inter_channels, -1)
        theta_x2 = self.theta2(x2).view(batch_size, self.inter_channels, -1)
        phi_x1 = self.phi1(x1).view(batch_size, self.inter_channels, -1)
        phi_x2 = self.phi2(x2).view(batch_size, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        theta_x2 = theta_x2.permute(0, 2, 1)
        f1 = torch.matmul(theta_x1, phi_x1)
        f2 = torch.matmul(theta_x2, phi_x2)

        f_div_C1 = F.softmax(f1, dim=-1)
        f_div_C2 = F.softmax(f2, dim=-1)
        y1 = torch.matmul(f_div_C2, g_x1)
        y2 = torch.matmul(f_div_C1, g_x2)
        y1 = y1.permute(0, 2, 1).contiguous()
        y2 = y2.permute(0, 2, 1).contiguous()
        y1 = y1.view(batch_size, self.inter_channels, *x1.size()[2:])
        y2 = y2.view(batch_size, self.inter_channels, *x1.size()[2:])

        W_y1 = self.W_z(y1)
        W_y2 = self.W_z2(y2)
        # residual connection
        z = W_y1 + W_y2
        return z

class NLBlock4(BaseModule):
    def __init__(self, in_channels):
        super(NLBlock4, self).__init__()
        self.inter_channels = in_channels // 2
        self.g = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.g2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta1 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.theta2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi1 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.phi2 = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.W_z = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.W_z2 = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x1 = self.g(x1).view(batch_size, self.inter_channels, -1)
        g_x2 = self.g2(x2).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x1 = self.theta1(x1).view(batch_size, self.inter_channels, -1)
        theta_x2 = self.theta2(x2).view(batch_size, self.inter_channels, -1)
        phi_x1 = self.phi1(x1).view(batch_size, self.inter_channels, -1)
        phi_x2 = self.phi2(x2).view(batch_size, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        theta_x2 = theta_x2.permute(0, 2, 1)
        f1 = torch.matmul(theta_x1, phi_x1)
        f2 = torch.matmul(theta_x2, phi_x2)

        f_div_C1 = F.softmax(f1, dim=-1)
        f_div_C2 = F.softmax(f2, dim=-1)
        y1 = torch.matmul(f_div_C2, g_x1)
        y2 = torch.matmul(f_div_C1, g_x2)
        y1 = y1.permute(0, 2, 1).contiguous()
        y2 = y2.permute(0, 2, 1).contiguous()
        y1 = y1.view(batch_size, self.inter_channels, *x1.size()[2:])
        y2 = y2.view(batch_size, self.inter_channels, *x1.size()[2:])

        W_y1 = self.W_z(y1)
        W_y2 = self.W_z2(y2)
        # residual connection
        z = x1 + W_y1 - W_y2 - x2
        return z

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            ConvModule(
                in_channels=in_features,
                out_channels=hidden_features,
                kernel_size=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
        )
        self.fc2 = nn.Sequential(
            ConvModule(
                in_channels=hidden_features,
                out_channels=out_features,
                kernel_size=1,
                norm_cfg=dict(type='BN'),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class PAB(BaseModule):
    def __init__(self,
                 out_channels=64, ):
        super(PAB, self).__init__()
        self.g_conv = nn.ModuleList()
        for i in range(1):
            self.g_conv.append(
                ConvModule(
                    in_channels=out_channels * 3,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=out_channels,
                ))
        self.downsample = nn.Sequential(nn.MaxPool2d(3, 2, 1))
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(scale_factor=4)

    @staticmethod
    def _concat(x1, x2, x3):
        return torch.cat((torch.cat((x1, x2), 1), x3), 1)

    def forward(self, sub):
        outs = []
        outs.append(self._concat(sub[0], self.upsample2(sub[1]), self.upsample4(sub[2])))
        outs[0] = channel_shuffle(outs[0], 3)
        outs[0] = self.g_conv[0](outs[0])
        outs.append(self.downsample(outs[0]))
        outs.append(self.downsample(outs[1]))
        return outs


class CAB(BaseModule):
    def __init__(self,
                 out_channels=64 * 1, ):
        super(CAB, self).__init__()
        self.g_conv1_1x1 = nn.ModuleList()
        self.g_conv2_1x1 = nn.ModuleList()
        self.res_layer = nn.ModuleList()
        # self.upsample = nn.ModuleList()
        self.last_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        for i in range(3):
            self.g_conv1_1x1.append(
                ConvModule(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=out_channels,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ))
            self.g_conv2_1x1.append(
                ConvModule(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=out_channels,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ))
            if i == 0:
                continue
            else:
                # self.upsample.append(
                #     nn.Upsample(scale_factor=2 ** i)
                # )
                self.res_layer.append(self.make_res_layer(
                    block=BasicBlock,
                    num_blocks=2,
                    in_channels=out_channels * 2,
                    out_channels=out_channels * 2,
                    expansion=1,
                    stride=2,
                    dilation=1,
                    style='pytorch',
                    avg_down=False,
                    norm_cfg=dict(type='BN'),
                    drop_path_rate=0.0))

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def forward(self, lab, x1, x2):
        outs1 = []
        outs2 = []
        outs = []
        for i in range(3):
            outs1.append(cat([lab[i], x1[i]], dim=1))
            outs1[i] = channel_shuffle(outs1[i], 2)
            outs1[i] = self.g_conv1_1x1[i](outs1[i])
            outs2.append(cat([lab[i], x2[i]], dim=1))
            outs2[i] = channel_shuffle(outs2[i], 2)
            outs2[i] = self.g_conv2_1x1[i](outs2[i])
            outs.append(cat([outs1[i], outs2[i]], dim=1))
        outs[1] = self.res_layer[0](outs[0]) + outs[1]
        outs[2] = self.res_layer[1](outs[1]) + outs[2]
        # outs[3] = self.res_layer[2](outs[2]) + outs[3]
        # out = outs[0] + self.upsample[0](outs[1]) + self.upsample[1](outs[2])
        return outs


@BACKBONES.register_module()
class ResNet_Graph_PCAM(ResNet):
    def __init__(self, mid_channels=64, **kwargs):
        super(ResNet_Graph_PCAM, self).__init__(**kwargs)
        self.mid_channels = mid_channels
        self.neck = nn.ModuleList([PAB(), CAB()])
        self.NLBlock = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(3):
            self.conv1x1.append(ConvModule(
                in_channels=mid_channels * 2 ** (i + 1),
                out_channels=64 * 1,
                kernel_size=1,
            ))
            self.NLBlock.append(NLBlock4(mid_channels))

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

        sub = []
        res_outs1 = []
        res_outs2 = []
        for t, layer_name in enumerate(self.res_layers):
            i = t - 1
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            if t == 0:
                continue
            else:
                res_outs1.append(self.conv1x1[i](x1))
                res_outs2.append(self.conv1x1[i](x2))
                sub.append(self.NLBlock[i](res_outs1[i], res_outs2[i]))
                # sub.append(self.conv1x1[i](x1) - self.conv1x1[i](x2))

        sub = self.neck[0](sub)
        x = self.neck[1](sub, res_outs1, res_outs2)
        return x
