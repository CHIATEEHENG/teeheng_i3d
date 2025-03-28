# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def extract_feat(self, inputs: Tensor, stage: str = 'neck', **kwargs):
        """Extract features from different model stages."""
        x = self.backbone(inputs)
        if self.with_neck:
           x, _ = self.neck(x)
        return x

