_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained='https://download.openmmlab.com/mmclassification/v0/conformer/conformer-small-p16_16xb128_in1k_20211016-cd6e7b89.pth'
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='Conformer',
        embed_dim=384,
        depth=12,
        patch_size=16,
        channel_ratio=4,
        num_heads=6,
        norm_eval=True,
        frozen_stages=0,
        out_indices=(4, 8, 11, 12),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_outs=5))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(max_epochs=12)

# add `find_unused_parameters=True` to avoid the error that the params not used in the detection
find_unused_parameters=True
