_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/master.py'
]


img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'img_norm_cfg', 'ori_filename'
                ]),
        ])
]

dataset_type = 'OCRDataset'
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file1 = 'tests/data/ocr_toy_dataset/label.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=100,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train_anno_file2 = 'tests/data/ocr_toy_dataset/label.lmdb'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='LmdbLoader',
        repeat=100,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

test_anno_file1 = 'tests/data/ocr_toy_dataset/label.lmdb'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    dataset_info='toy_dataset',
    test_mode=True)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1, train2]),
    val=dict(type='ConcatDataset', datasets=[test, test]),
    test=dict(type='ConcatDataset', datasets=[test]))

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[50, 70])
total_epochs = 100

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
# find_unused_parameters = True