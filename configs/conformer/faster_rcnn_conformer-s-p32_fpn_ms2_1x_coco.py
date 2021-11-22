_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained='https://download.openmmlab.com/mmclassification/v0/conformer/conformer-small-p32_16xb128_in1k_20211016-5756a4d3.pth'
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='Conformer',
        embed_dim=384,
        depth=12,
        patch_size=32,
        channel_ratio=4,
        num_heads=6,
	drop_path_rate=0.1,
        norm_eval=True,
        frozen_stages=0,
        out_indices=(4, 8, 11, 12),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_outs=5))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.),
            'bn': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)

# add `find_unused_parameters=True` to avoid the error that the params not used in the detection
find_unused_parameters=True
