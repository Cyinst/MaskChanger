import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def main():
    # args = parse_args()
    # cfg = mmcv.Config.fromfile('work_dirs/graph_pcam_r18_512x512_40k_levircd_90.17/graph_pcam_r18_512x512_40k_levircd.py')
    # # build the model from a config file and a checkpoint file
    # model = init_segmentor(cfg, device='cuda:0')
    # if args.preview_model:
    #     print(model)
    # cfg = mmcv.Config.fromfile(args.config)
    cfg = mmcv.Config.fromfile('work_dirs/graph_pcam_r18_512x512_40k_levircd_90.17/graph_pcam_r18_512x512_40k_levircd.py')
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # print('here')
    print(model)

main()
