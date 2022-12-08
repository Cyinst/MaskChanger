# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='IAEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNeSt_PCAM',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        stem_channels=64,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        mid_channels=256),
    decode_head=dict(
        type='Seg_head',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))