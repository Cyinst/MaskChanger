_base_ = './pcam_r18_512x512_40k_levircd.py'

model = dict(
    backbone=dict(
        type='ResNet_Graph_PCAM',
    ),
    decode_head=dict(
        in_channels=[128, 128, 128],
        in_index=[0, 1, 2],),

    )
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,)

evaluation = dict(interval=1000, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])