import glob
import os
import os.path as osp

import mmcv
import torch.nn as nn

from mmaction.models import build_recognizer


def _get_config_path():
    """ Find the predefined recognizer config path."""
    repo_dir = osp.dirname(osp.dirname(__file__))
    config_dpath = osp.join(repo_dir, 'config')
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config_fpaths = list(glob.glob(osp.join(config_dpath, '*.py')))
    config_names = [os.path.relpath(p, config_dpath) for p in config_fpaths]
    print(f'Using {len(config_names)} config files')
    config_fpaths = [
        osp.join(config_dpath, config_fpath) for config_fpath in config_fpaths
    ]
    return config_fpaths


def test_config_build_recognizer():
    """
    Test that all mmaction models defined in the configs can be initialized.
    """
    config_fpaths = _get_config_path()

    # test all config file in `config` directory
    for config_fpath in config_fpaths:
        config_mod = mmcv.Config.fromfile(config_fpath)
        print(f'Building recognizer, config_fpath = {config_fpath!r}')

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model['backbone']:
            config_mod.model['backbone']['pretrained'] = None

        recognizer = build_recognizer(
            config_mod.model,
            train_cfg=config_mod.train_cfg,
            test_cfg=config_mod.test_cfg)
        assert isinstance(recognizer, nn.Module)
