"""
Model configuration templates for different SOLOv2 sizes
"""

MODEL_CONFIGS = {
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


def get_model_config(model_size='medium'):
    """Get model configuration by size"""
    if model_size not in MODEL_CONFIGS:
        print(f"Unknown model size: {model_size}. Using 'medium' instead.")
        model_size = 'medium'

    return MODEL_CONFIGS[model_size]


def list_available_models():
    """List all available model configurations"""
    print("\n" + "="*70)
    print("Available SOLOv2 Model Sizes:")
    print("="*70)
    for size, config in MODEL_CONFIGS.items():
        print(f"\n{size.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Backbone: {config['backbone']['type']}{config['backbone']['depth']}")
        print(f"  FPN Channels: {config['neck']['out_channels']}")
        print(f"  Default Batch Size: {config['batch_size']}")
    print("\n" + "="*70 + "\n")
