from mmseg.models.builder import BACKBONES
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, build_upsample_layer)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmseg.models.backbones import ResNet
import torch
import torch.nn as nn
from torch import cat
import torch.utils.checkpoint as cp

eps = 1.0e-5


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class BasicBlock(BaseModule):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        # elif issubclass(block, Bottleneck):
        #     expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class PAB_attentionmap(BaseModule):
    def __init__(self,
                 out_channels=64 * 4, ):
        super(PAB_attentionmap, self).__init__()
        self.g_conv = nn.ModuleList()
        for i in range(4):
            self.g_conv.append(
                ConvModule(
                    in_channels=out_channels * 4,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=out_channels,
                ))
        self.downsample = nn.Sequential(nn.MaxPool2d(3, 2, 1))
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample8 = nn.Upsample(scale_factor=8)

    @staticmethod
    def _concat(x1, x2, x3, x4):
        return torch.cat((torch.cat((torch.cat((x1, x2), 1), x3), 1), x4), 1)

    def forward(self, sub):
        outs = []
        outs.append(self._concat(sub[0], self.upsample2(sub[1]), self.upsample4(sub[2]), self.upsample8(sub[3])))
        outs.append(self.downsample(outs[0]))
        outs.append(self.downsample(outs[1]))
        outs.append(self.downsample(outs[2]))
        for i in range(4):
            outs[i] = self.g_conv[i](outs[i])
        return outs


class CAB(BaseModule):
    def __init__(self,
                 out_channels=64 * 4, ):
        super(CAB, self).__init__()
        self.g_conv_1x1 = nn.ModuleList()
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
        for i in range(4):
            self.g_conv_1x1.append(
                ConvModule(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=out_channels,
                    norm_cfg=dict(type='BN'),
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
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion=1,
                    stride=2,
                    dilation=1,
                    style='pytorch',
                    avg_down=False,
                    norm_cfg=dict(type='BN'),
                    drop_path_rate=0.0))

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def forward(self, lab, x):
        outs = []
        for i in range(4):
            outs.append(cat([lab[i], x[i]], dim=1))
            outs[i] = channel_shuffle(outs[i], 2)
            outs[i] = self.g_conv_1x1[i](outs[i])
        outs[1] = self.res_layer[0](outs[0]) + outs[1]
        outs[2] = self.res_layer[1](outs[1]) + outs[2]
        outs[3] = self.res_layer[2](outs[2]) + outs[3]
        # out = outs[0] + self.upsample[0](outs[1]) + self.upsample[1](outs[2])
        return outs


class CAB_multiple(BaseModule):
    def __init__(self,
                 out_channels=64 * 4, ):
        super(CAB_multiple, self).__init__()
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
            if i == 0:
                continue
            else:
                # self.upsample.append(
                #     nn.Upsample(scale_factor=2 ** i)
                # )
                self.res_layer.append(self.make_res_layer(
                    block=BasicBlock,
                    num_blocks=2,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion=1,
                    stride=2,
                    dilation=1,
                    style='pytorch',
                    avg_down=False,
                    norm_cfg=dict(type='BN'),
                    drop_path_rate=0.0))

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def forward(self, pab, x):
        outs = []
        for i in range(3):
            outs.append(pab[i] * x[i])
        outs[1] = self.res_layer[0](outs[0]) + outs[1]
        outs[2] = self.res_layer[1](outs[1]) + outs[2]
        return self.last_conv(outs[2])


@BACKBONES.register_module()
class ResNet_PAB_CAB(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_PAB_CAB, self).__init__(**kwargs)
        self.neck = nn.ModuleList([PAB_attentionmap(), CAB()])
        self.conv1x1 = nn.ModuleList()
        for i in range(4):
            self.conv1x1.append(ConvModule(
                in_channels=64 * 2 ** i,
                out_channels=256,
                kernel_size=1,
            ))

    def forward(self, x1, x2, interact_layer=[1, 2, 3]):
        x1 = self.conv1(x1)
        x1 = self.norm1(x1)
        x1 = self.relu(x1)
        x2 = self.conv1(x2)
        x2 = self.norm1(x2)
        x2 = self.relu(x2)
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)

        x = []
        res_outs = []
        for i, layer_name in enumerate(self.res_layers):
            # if i > 2:
            #     break
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            x.append(self.conv1x1[i](x1) - self.conv1x1[i](x2))
            res_outs.append(self.conv1x1[i](x1))

        x = self.neck[0](x)
        x = self.neck[1](x, res_outs)
        return x


# ablation
class PAB_without_channelshuffle(BaseModule):
    def __init__(self,
                 out_channels=64 * 4, ):
        super(PAB_without_channelshuffle, self).__init__()
        self.planes = 64
        self.g_conv = nn.ModuleList()
        for i in range(3):
            self.g_conv.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # groups=out_channels,
                ))
        self.downsample1to2 = nn.Sequential(nn.MaxPool2d(3, 2, 1))
        self.downsample1to3 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1)
        )
        self.downsample2to3 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1))
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(scale_factor=4)

    @staticmethod
    def _concat(x1, x2, x3):
        return torch.cat((torch.cat((x1, x2), 1), x3), 1)

    def forward(self, sub):
        out = []
        for i in range(3):
            out.append(self.g_conv[i](sub[i]))
        return out


@BACKBONES.register_module()
class ResNet_PAB_CAB_ablation(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_PAB_CAB_ablation, self).__init__(**kwargs)
        # self.neck = nn.ModuleList([PAB_without_channelshuffle(), CAB_bottleneck()])
        # self.neck = nn.ModuleList([CAB()])
        # self.conv1x1 = nn.ModuleList()
        # for i in range(3):
        #     self.conv1x1.append(ConvModule(
        #         in_channels=64 * 2 ** i,
        #         out_channels=256,
        #         kernel_size=1,
        #     ))
        self.conv1x1 = ConvModule(
            in_channels=64 * 8,
            out_channels=256,
            kernel_size=1,
        )

    def forward(self, x):
        x_width = len(x[0][0][0])
        x1 = x[:, :, :, :x_width // 2]
        x2 = x[:, :, :, x_width // 2:]
        x1 = self.conv1(x1)
        x1 = self.norm1(x1)
        x1 = self.relu(x1)
        x2 = self.conv1(x2)
        x2 = self.norm1(x2)
        x2 = self.relu(x2)
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)

        x = []
        res_outs = []
        for i, layer_name in enumerate(self.res_layers):
            if i > 2:
                break
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            # x.append(self.conv1x1[i](x2))
            # x.append(self.conv1x1[i](x1) - self.conv1x1[i](x2))
            # res_outs.append(self.conv1x1[i](x1))

        # x = self.neck[0](x)
        # x = self.neck[0](x, res_outs)
        # x = self.neck[1](x, res_outs)
        return self.conv1x1(cat([x1, x2], dim=1))
