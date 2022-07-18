_base_ = [
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# model settings
num_classes = 313
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet101',
        depth=101,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=num_classes,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        loss_cls=dict(type='BCELossWithLogits', loss_weight=160.0),
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        label_smooth_eps=0),
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/mmit/videos'
data_root_val = '/data/mmit/videos'
ann_file_train = 'data/mmit/mmit_train_list_videos.txt'
ann_file_val = 'data/mmit/mmit_val_list_videos.txt'
ann_file_test = 'data/mmit/mmit_val_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=313))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=313,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=313,
        test_mode=True))

val_evaluator = dict(
    type='AccMetric',
    metrics=['mmit_mean_average_precision'],
    num_classes=num_classes,
    prefix='prec',
)
test_evaluator = val_evaluator

val_cfg = dict(interval=5)
test_cfg = dict()

default_hooks = dict(checkpoint=dict(interval=5))
