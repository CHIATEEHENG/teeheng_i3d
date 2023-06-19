_base_ = '../../_base_/default_runtime.py'

# model settings
model = dict(
    type='TCANet',
    feat_dim=2048,
    se_sample_num=32,
    action_sample_num=64,
    temporal_dim=100,
    window_size=9,
    lgte_num=2,
    soft_nms_alpha=0.4,
    soft_nms_low_threshold=0.0,
    soft_nms_high_threshold=0.0,
    post_process_top_k=100,
    feature_extraction_interval=16)

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/HACS/slowonly_feature/'
data_root_val = 'data/HACS/slowonly_feature/'
ann_file_train = 'data/HACS/hacs_anno_train.json'
ann_file_val = 'data/HACS/hacs_anno_val.json'
ann_file_test = 'data/HACS/hacs_anno_val.json'

train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', ),
        meta_keys=('video_name', ))
]

val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', ),
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame'))
]

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', ),
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

max_epochs = 9
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[
            7,
        ],
        gamma=0.1)
]

work_dir = './work_dirs/tcanet_2xb8-2048x100-9e_hacs-feature/'
test_evaluator = dict(
    type='ANetMetric',
    metric_type='AR@AN',
    dump_config=dict(out=f'{work_dir}/results.json', output_format='json'))
val_evaluator = test_evaluator
