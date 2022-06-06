_base_ = [
    '../../_base_/models/slowonly_r50.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(pretrained=None, in_channels=2, with_pool2=False))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/gym/rawframes'
data_root_val = 'data/gym/rawframes'
ann_file_train = 'data/gym/annotations/gym99_train_list_rawframes.txt'
ann_file_val = 'data/gym/annotations/gym99_val_list_rawframes.txt'
ann_file_test = 'data/gym/annotations/gym99_val_list_rawframes.txt'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128])
train_pipeline = [
    dict(type='SampleFrames', clip_len=4, frame_interval=16, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=24,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=24,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=val_pipeline,
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
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

val_cfg = dict(interval=5)
test_cfg = dict()

# optimizer
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=120,
        by_epoch=True,
        milestones=[90, 110],
        gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=120)
# runtime settings

load_from = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'
             'slowonly_r50_4x16x1_256e_kinetics400_flow/'
             'slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth')

default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=40, norm_type=2)))
