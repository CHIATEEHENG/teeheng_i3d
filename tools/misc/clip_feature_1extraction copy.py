# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import torch
from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 feature extraction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_prefix', type=str, help='output prefix')
    parser.add_argument('--video-list', type=str, default=None, help='video file list')
    parser.add_argument('--video-root', type=str, default=None, help='video root directory')
    parser.add_argument('--spatial-type', type=str, default='avg', choices=['avg', 'max', 'keep'], help='Pooling type in spatial dimension')
    parser.add_argument('--temporal-type', type=str, default='avg', choices=['avg', 'max', 'keep'], help='Pooling type in temporal dimension')
    parser.add_argument('--long-video-mode', action='store_true', help='Perform long video inference')
    parser.add_argument('--clip-interval', type=int, default=None, help='Clip interval for long video inference')
    parser.add_argument('--frame-interval', type=int, default=None, help='Temporal interval for long video inference')
    parser.add_argument('--multi-view', action='store_true', help='Perform multi-view inference')
    parser.add_argument('--dump-score', action='store_true', help='Dump prediction scores rather than features')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override some settings in the used config')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='Job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args

def merge_args(cfg, args):
    """Modify config to extract features instead of classification predictions."""
    if cfg.model.get("cls_head", None):
        cfg.model.cls_head = None  # ✅ Remove classification head

    cfg.model.test_cfg = dict(feature_extraction=True)  # ✅ Enable feature extraction mode

    return cfg

def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher

    # ✅ Ensure work_dir is set
    cfg.work_dir = args.output_prefix  

    cfg.load_from = args.checkpoint

    # ✅ Remove classification head from model
    cfg.model.cls_head = None  

    # ✅ Build the runner
    runner = Runner.from_cfg(cfg)

    # ✅ Extract features
    all_features = []
    for data_batch in runner.test_dataloader:
        batch_inputs = torch.stack([torch.tensor(inp, dtype=torch.float32) / 255.0 for inp in data_batch['inputs']])
        batch_features, _ = runner.model.extract_feat(batch_inputs)
        all_features.append(batch_features.detach().cpu().numpy())

    # ✅ Reshape Correctly: (batch, num_clips, 2048, ...) → (2048, num_frames)
    all_features = np.concatenate(all_features, axis=0)  # Flatten batch
    all_features = all_features.reshape(2048, -1)  # Ensure shape (2048, num_frames)

    # ✅ Save as `.npy` (not `.npz`)
    output_path = osp.join(args.output_prefix, "features.npy")
    np.save(output_path, all_features.astype(np.float32))

    print(f"✅ Features successfully extracted and saved to {output_path}")

if __name__ == '__main__':
    main()
