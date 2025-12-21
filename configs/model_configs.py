"""
Model configuration templates for different SOLOv2 and RTMDet-Ins sizes
"""

SOLOV2_CONFIGS = {
    'nano': {
        'backbone': {
            'type': 'ResNet',
            'depth': 18,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'frozen_stages': 1,
            'init_cfg': dict(type='Pretrained', checkpoint='torchvision://resnet18'),
            'style': 'pytorch'
        },
        'neck': {
            'type': 'FPN',
            'in_channels': [64, 128, 256, 512],
            'out_channels': 128,
            'start_level': 0,
            'num_outs': 5
        },
        'mask_head': {
            'in_channels': 128,
            'feat_channels': 256,
            'stacked_convs': 2,
            'mask_feature_head': {
                'feat_channels': 64,
                'start_level': 0,
                'end_level': 3,
                'out_channels': 128,
                'mask_stride': 4,
                'norm_cfg': dict(type='GN', num_groups=32, requires_grad=True)
            }
        },
        'batch_size': 8,
        'description': 'Nano - ResNet18 backbone, fastest inference, lowest memory'
    },
    'small': {
        'backbone': {
            'type': 'ResNet',
            'depth': 34,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'frozen_stages': 1,
            'init_cfg': dict(type='Pretrained', checkpoint='torchvision://resnet34'),
            'style': 'pytorch'
        },
        'neck': {
            'type': 'FPN',
            'in_channels': [64, 128, 256, 512],
            'out_channels': 192,
            'start_level': 0,
            'num_outs': 5
        },
        'mask_head': {
            'in_channels': 192,
            'feat_channels': 384,
            'stacked_convs': 3,
            'mask_feature_head': {
                'feat_channels': 96,
                'start_level': 0,
                'end_level': 3,
                'out_channels': 192,
                'mask_stride': 4,
                'norm_cfg': dict(type='GN', num_groups=32, requires_grad=True)
            }
        },
        'batch_size': 6,
        'description': 'Small - ResNet34 backbone, balanced speed/accuracy'
    },
    'medium': {
        'backbone': {
            'type': 'ResNet',
            'depth': 50,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'frozen_stages': 1,
            'init_cfg': dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            'style': 'pytorch'
        },
        'neck': {
            'type': 'FPN',
            'in_channels': [256, 512, 1024, 2048],
            'out_channels': 256,
            'start_level': 0,
            'num_outs': 5
        },
        'mask_head': {
            'in_channels': 256,
            'feat_channels': 512,
            'stacked_convs': 4,
            'mask_feature_head': {
                'feat_channels': 128,
                'start_level': 0,
                'end_level': 3,
                'out_channels': 256,
                'mask_stride': 4,
                'norm_cfg': dict(type='GN', num_groups=32, requires_grad=True)
            }
        },
        'batch_size': 4,
        'description': 'Medium - ResNet50 backbone, good accuracy (default)'
    },
    'large': {
        'backbone': {
            'type': 'ResNet',
            'depth': 101,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'frozen_stages': 1,
            'init_cfg': dict(type='Pretrained', checkpoint='torchvision://resnet101'),
            'style': 'pytorch'
        },
        'neck': {
            'type': 'FPN',
            'in_channels': [256, 512, 1024, 2048],
            'out_channels': 384,
            'start_level': 0,
            'num_outs': 5
        },
        'mask_head': {
            'in_channels': 384,
            'feat_channels': 768,
            'stacked_convs': 5,
            'mask_feature_head': {
                'feat_channels': 192,
                'start_level': 0,
                'end_level': 3,
                'out_channels': 384,
                'mask_stride': 4,
                'norm_cfg': dict(type='GN', num_groups=32, requires_grad=True)
            }
        },
        'batch_size': 2,
        'description': 'Large - ResNet101 backbone, best accuracy, slower'
    }
}

RTMDET_INS_CONFIGS = {
    'tiny': {
        'backbone': {
            'type': 'CSPNeXt',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 0.167,
            'widen_factor': 0.375,
            'channel_attention': True,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True),
            'init_cfg': dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
            )
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': [96, 192, 384],
            'out_channels': 96,
            'num_csp_blocks': 1,
            'expand_ratio': 0.5,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True)
        },
        'bbox_head': {
            'type': 'RTMDetInsSepBNHead',
            'in_channels': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'share_conv': True,
            'pred_kernel_size': 1,
            'act_cfg': dict(type='SiLU', inplace=True),
            'norm_cfg': dict(type='SyncBN', requires_grad=True)
        },
        'batch_size': 10,
        'description': 'Tiny - CSPNeXt tiny backbone, fastest inference, lowest memory'
    },
    'small': {
        'backbone': {
            'type': 'CSPNeXt',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 0.33,
            'widen_factor': 0.5,
            'channel_attention': True,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True),
            'init_cfg': dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
            )
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': [128, 256, 512],
            'out_channels': 128,
            'num_csp_blocks': 1,
            'expand_ratio': 0.5,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True)
        },
        'bbox_head': {
            'type': 'RTMDetInsSepBNHead',
            'in_channels': 128,
            'feat_channels': 128,
            'stacked_convs': 2,
            'share_conv': True,
            'pred_kernel_size': 1,
            'act_cfg': dict(type='SiLU', inplace=True),
            'norm_cfg': dict(type='SyncBN', requires_grad=True)
        },
        'batch_size': 8,
        'description': 'Small - CSPNeXt small backbone, balanced speed/accuracy'
    },
    'medium': {
        'backbone': {
            'type': 'CSPNeXt',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 0.67,
            'widen_factor': 0.75,
            'channel_attention': True,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True),
            'init_cfg': dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_imagenet_600e.pth'
            )
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': [192, 384, 768],
            'out_channels': 192,
            'num_csp_blocks': 2,
            'expand_ratio': 0.5,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True)
        },
        'bbox_head': {
            'type': 'RTMDetInsSepBNHead',
            'in_channels': 192,
            'feat_channels': 192,
            'stacked_convs': 2,
            'share_conv': True,
            'pred_kernel_size': 1,
            'act_cfg': dict(type='SiLU', inplace=True),
            'norm_cfg': dict(type='SyncBN', requires_grad=True)
        },
        'batch_size': 6,
        'description': 'Medium - CSPNeXt medium backbone, good accuracy'
    },
    'large': {
        'backbone': {
            'type': 'CSPNeXt',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 1.0,
            'widen_factor': 1.0,
            'channel_attention': True,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True),
            'init_cfg': dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_imagenet_600e.pth'
            )
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': [256, 512, 1024],
            'out_channels': 256,
            'num_csp_blocks': 3,
            'expand_ratio': 0.5,
            'norm_cfg': dict(type='SyncBN'),
            'act_cfg': dict(type='SiLU', inplace=True)
        },
        'bbox_head': {
            'type': 'RTMDetInsSepBNHead',
            'in_channels': 256,
            'feat_channels': 256,
            'stacked_convs': 2,
            'share_conv': True,
            'pred_kernel_size': 1,
            'act_cfg': dict(type='SiLU', inplace=True),
            'norm_cfg': dict(type='SyncBN', requires_grad=True)
        },
        'batch_size': 4,
        'description': 'Large - CSPNeXt large backbone, best accuracy, slower'
    }
}


def get_model_config(model_size='medium', architecture='solov2'):
    """Get model configuration by size and architecture"""
    configs = SOLOV2_CONFIGS if architecture == 'solov2' else RTMDET_INS_CONFIGS

    if model_size not in configs:
        print(f"Unknown model size: {model_size}. Using 'medium' for {architecture}.")
        model_size = 'medium' if model_size in ['medium', 'large'] else 'small'

    # Handle 'nano' for RTMDet-Ins (uses 'tiny' instead)
    if architecture == 'rtmdet-ins' and model_size == 'nano':
        print(f"RTMDet-Ins doesn't have 'nano' size. Using 'tiny' instead.")
        model_size = 'tiny'

    return configs[model_size]


def list_available_models():
    """List all available model configurations"""
    print("\n" + "="*70)
    print("Available Model Sizes:")
    print("="*70)

    print("\nSOLOv2 Models:")
    print("-" * 70)
    for size, config in SOLOV2_CONFIGS.items():
        print(f"\n{size.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Backbone: {config['backbone']['type']}{config['backbone']['depth']}")
        print(f"  FPN Channels: {config['neck']['out_channels']}")
        print(f"  Default Batch Size: {config['batch_size']}")

    print("\n" + "-" * 70)
    print("\nRTMDet-Ins Models:")
    print("-" * 70)
    for size, config in RTMDET_INS_CONFIGS.items():
        print(f"\n{size.upper()}:")
        print(f"  Description: {config['description']}")
        backbone_info = f"{config['backbone']['type']} (deepen={config['backbone']['deepen_factor']}, widen={config['backbone']['widen_factor']})"
        print(f"  Backbone: {backbone_info}")
        print(f"  Neck Channels: {config['neck']['out_channels']}")
        print(f"  Default Batch Size: {config['batch_size']}")

    print("\n" + "="*70 + "\n")
