_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/master.py'
]


img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromLMDB'),
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
    dict(type='LoadImageFromLMDB'),
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
img_prefix = ''
train_anno_file1 = '/synth_data/data1'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train_anno_file2 = '/synth_data/data2'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train_anno_file3 = '/real_data/data1'
train3 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train_anno_file4 = '/real_data/data2'
train4 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

# synth data
synth_data = dict(
    type='WeightedConcatDataset',
    datasets=[train1, train2],
    weights=[1, 1],
    len_epoch=10000,
)

# real data
real_data = dict(
    type='WeightedConcatDataset',
    datasets=[train3, train4],
    weights=[1, 1],
    len_epoch=10000,
)

test_ann_files = {'CUTE80':'/mjsynth_training/evaluation/CUTE80',
                  'IC03_860':'/mjsynth_training/evaluation/IC03_860',
                  'IC03_867':'/mjsynth_training/evaluation/IC03_867',
                  'IC13_1015':'/mjsynth_training/evaluation/IC13_1015',
                  'IC13_857':'/mjsynth_training/evaluation/IC13_857',
                  'IC15_1811':'/mjsynth_training/evaluation/IC15_1811',
                  'IC15_2077':'/mjsynth_training/evaluation/IC15_2077',
                  'iiit5k':'/mjsynth_training/iiit5k/',
                  'SVT':'/mjsynth_training/evaluation/SVT',
                  'SVTP':'/mjsynth_training/evaluation/SVTP'}

testset = [dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    dataset_info=dataset_name,
    test_mode=True) for dataset_name, test_ann_file in test_ann_files.items()]

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=2,
    # In real project develop, we should keep balance between synth_data and real_data (9:1 in one batch.)
    train=dict(type='WeightedConcatDataset', datasets=[synth_data, real_data], weights=[9, 1], len_epoch=10000),
    val=dict(type='ConcatDataset', datasets=testset),
    test=dict(type='ConcatDataset', datasets=testset))

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
    step=[8, 13])
total_epochs = 15

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=1000,
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