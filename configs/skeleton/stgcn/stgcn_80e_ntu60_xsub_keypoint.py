_base_ = [
    '../../_base_/models/stgcn.py',
    '../../_base_/default_runtime.py'
]

preprocess_cfg = dict(
    mean=[960., 540., 0.5],
    std=[1920, 1080, 1.],
    format_shape='NCTVM')

model = dict(
    backbone=dict(
        graph_cfg=dict(layout='coco', strategy='spatial')),
    data_preprocessor=dict(
        type='ActionDataPreprocessor', **preprocess_cfg))

dataset_type = 'PoseDataset'
ann_file_train = 'data/posec3d/ntu60_xsub_train.pkl'
ann_file_val = 'data/posec3d/ntu60_xsub_val.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_train, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=test_pipeline, test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=test_pipeline, test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_begin=1,
    val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[10, 50],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                   nesterov=True))

default_hooks = dict(checkpoint=dict(interval=3))
