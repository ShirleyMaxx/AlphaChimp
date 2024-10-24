_base_ = '../_base_/default_runtime.py'

num_levels = 5
model = dict(
    type='mmaction.AlphaChimp',
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/alphachimp/pretrain_res576.pth'),
    model_cfg=dict(
        type='mmdet.DINO',
        num_feature_levels=num_levels,
        num_queries=10,  # num_matching_queries
        with_box_refine=True,
        as_two_stage=True,
        data_preprocessor=None,
        backbone=dict(
            type='mmaction.SwinTransformer3D',
            arch='large',
            pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_large_patch4_window7_224_22k.pth',
            pretrained2d=True,
            patch_size=(2, 4, 4),
            window_size=(8, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.40,
            out_indices=(0, 1, 2, 3),
            patch_norm=True),
        neck=dict(
            type='mmdet.ChannelMapper',
            in_channels=[192, 384, 768, 1536],
            kernel_size=1,
            tmp_stride=4,
            out_channels=512,
            norm_cfg=dict(type='GroupNorm', num_groups=32),
            num_outs=num_levels),
        encoder=dict(
            num_layers=12,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=512, num_levels=num_levels,
                                   dropout=0.20),  # 0.1 for DeformDETR
                ffn_cfg=dict(
                    embed_dims=512,
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_drop=0.20))),  # 0.1 for DeformDETR
        decoder=dict(
            num_layers=12,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=512, num_heads=16,
                                   dropout=0.20),  # 0.1 for DeformDETR
                cross_attn_cfg=dict(embed_dims=512, num_levels=num_levels,
                                    dropout=0.20),  # 0.1 for DeformDETR
                ffn_cfg=dict(
                    embed_dims=512,
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_drop=0.20)),  # 0.1 for DeformDETR
            post_norm_cfg=None),
        positional_encoding=dict(
            num_feats=256,
            normalize=True,
            offset=-0.5,  # -0.5 for DeformDETR
            temperature=10000),  # 10000 for DeformDETR
        bbox_head=dict(
            type='mmdet.DINOHead',
            embed_dims=512,
            num_classes=24,
            mlp_cls=True,
            sync_cls_avg_factor=True,
            loss_cls=dict(type='MultilableCrossEntropy', mask_cls=False, no_obj_mode=True, loss_weight=2.0),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
            loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0)),
        dn_cfg=dict(  # TODO: Move to model.train_cfg ?
            label_noise_scale=0.00001,
            box_noise_scale=0.2,  # 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None,
                           num_dn_queries=2)),  # TODO: half num_dn_queries
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.FocalLossCost', weight=3.0, cls_weight=0.0, obj_weight=1.0, no_obj_mode=True),
                    dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0)
                ],
                num_classes=24)),
        test_cfg=dict(max_per_img=10)),  # 100 for DeformDETR
    data_preprocessor=dict(
        type='mmaction.MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs = dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCTHW'),
            proposals = dict(
                type='NoDataPreprocessor',
            )),),
    train_cfg=None,
    test_cfg=None)


dataset_type = 'LeipzigChimpActionDatasetOri'
data_root = 'data/ChimpACT_processed'
anno_root = 'data/ChimpACT_processed/annotations/action'

ann_file_train = f'{anno_root}/train_action.csv'
ann_file_val = f'{anno_root}/test_action.csv'


exclude_file_train = f'{anno_root}/train_action_excluded_timestamps.csv'
exclude_file_val = f'{anno_root}/test_action_excluded_timestamps.csv'


label_file = f'{anno_root}/action_list.txt'

proposal_file_train = (f'{anno_root}/train_action_gt.pkl')
proposal_file_val = f'{anno_root}/test_action_gt.pkl'


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=((576, 576), (760, 760))),
    dict(type='RandomCrop', size=576),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RandomErasing', erase_prob=0.15, min_area_ratio=0.05, max_area_ratio=0.10),
    dict(type='ColorJitter', brightness=0.35, contrast=0.35, saturation=0.35, hue=0.10),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', collect_keys=['imgs'],padding=True,max_person=24)
]

# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(576, 576), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', collect_keys=['imgs'],padding=True,max_person=24)
]


train_dataloader = dict(
    batch_size=2,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    num_classes=24,
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val,
    is_dino=True,
    #dino_thr=0.5,
    #dino_act_thr=0.5,
    )

test_evaluator = val_evaluator

#train_cfg = dict(
#    type='EpochBasedTrainLoop', max_epochs=80, val_begin=1, val_interval=1)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=15000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
)

log_processor = dict(
    by_epoch=False
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0005),
    accumulative_counts=4,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.5)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=False,
        milestones=[24000, 48000, 64000],
        gamma=0.2)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

#randomness = dict(seed=247161711, deterministic=True)