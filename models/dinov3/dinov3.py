import torch
import timm
from torch import nn
from collections import OrderedDict
from peft import LoraConfig, get_peft_model


def _freeze_backbone_layers(model, args):
    if args is None or not hasattr(args, 'freeze_backbone_layers'):
        return
    freeze_layers = args.freeze_backbone_layers
    if freeze_layers is None:
        return
    if freeze_layers == -1:
        print("Freezing entire backbone (all layers)...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        return
    elif freeze_layers > 0:
        print(f"Freezing first {freeze_layers} layers of backbone...")

        if model.is_vit:
            model.backbone.cls_token.requires_grad = False
            model.backbone.reg_token.requires_grad = False

            for param in model.backbone.patch_embed.parameters():
                param.requires_grad = False

            num_blocks = len(model.backbone.blocks)
            freeze_count = min(freeze_layers, num_blocks)
            for i in range(freeze_count):
                for param in model.backbone.blocks[i].parameters():
                    param.requires_grad = False

            print(f"  ✓ Froze: cls_token, reg_token, patch_embed")
            print(f"  ✓ Froze: blocks[0:{freeze_count}] (total {num_blocks} blocks)")
            print(f"  ✓ Trainable: blocks[{freeze_count}:{num_blocks}], norm, head")

def _apply_lora_to_model(model, args):
    """
    Apply LoRA from PEFT library
    Args:
        model: DINOv3 model
        args: args for LoRA
    Returns:
        model with LoRA
    """
    if not hasattr(args, 'use_lora') or not args.use_lora:
        return model

    # Get args for LoRA
    lora_rank = getattr(args, 'lora_rank', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16.0)
    lora_dropout = getattr(args, 'lora_dropout', 0.1)
    lora_target_modules = getattr(args, 'lora_target_modules', None)

    # Select target_modules
    if lora_target_modules is None:
        if model.is_vit:
            # ViT: attention layer
            lora_target_modules = ["qkv", "proj"]
        else:
            # default
            lora_target_modules = ["qkv", "proj"]

    # configurate LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    print(f"\n{'=' * 60}")
    print("Applying LoRA with PEFT library...")
    print(f"  - Rank: {lora_rank}")
    print(f"  - Alpha: {lora_alpha}")
    print(f"  - Dropout: {lora_dropout}")
    print(f"  - Target modules: {lora_target_modules}")
    print(f"{'=' * 60}\n")

    model = get_peft_model(model, lora_config)
    print("Trainable parameters in LoRA model:")
    model.print_trainable_parameters()
    return model

class DINOv3(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 
        )

        self.embed_dim = self.backbone.num_features

        self.is_vit = 'vit' in model_name.lower()

        self.embedding_target_dim = 100
        if self.embed_dim > self.embedding_target_dim:
            print(f"Adding pooling for embeddings: {self.embed_dim} -> {self.embedding_target_dim}")
            self.embedding_pool = nn.AdaptiveAvgPool1d(self.embedding_target_dim)
        else:
            self.embedding_pool = None

        self.global_pool = None
        self.head = None
        self.reset_classifier(num_classes)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool = nn.Identity()

        fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

    def forward_features(self, x):
        """
        Get features for neighbor-smoothing (compute similarities)

        - ViT: (batch, num_patches+1, embed_dim), need CLS token
        """
        features = self.backbone.forward_features(x)

        # Check backbone
        if len(features.shape) == 3:
            # ViT : (batch, num_tokens, embed_dim)
            # The 0th token is the CLS token, which is used as the global feature.
            features = features[:, 0]

        elif len(features.shape) == 2:
            pass

        else:
            raise ValueError(
                f"Unexpected feature shape: {features.shape}. "
                f"Expected 2D (batch, embed_dim), 3D (batch, num_tokens, embed_dim), "
                f"or 4D (batch, channels, H, W)."
            )

        return features

    def forward_features_spatial(self, x):
        """Get spatial features for ML-Decoder"""
        features = self.backbone.forward_features(x)

        if self.is_vit:
            # ViT: remove CLS token，only keep patch tokens
            if len(features.shape) == 3:
                spatial_features = features[:, 1:, :]
            else:
                raise ValueError(f"Unexpected ViT output shape: {features.shape}")
        else:
            if len(features.shape) == 3:
                spatial_features = features
            elif len(features.shape) == 4:
                batch, channels, height, width = features.shape
                spatial_features = features.flatten(2).transpose(1, 2)
            else:
                raise ValueError(f"Cannot extract spatial features from shape: {features.shape}")

        return spatial_features

    def _forward_impl(self, x, return_embedding=False, return_label_features=False):
        if return_embedding:
            features = self.forward_features(x)
            if self.embedding_pool is not None:
                features = self.embedding_pool(features.unsqueeze(1)).squeeze(1)
            return features
        else:
            features = self.forward_features_spatial(x)
        if return_label_features:
            x = self.head(features, return_label_features=True)
        else:
            x = self.head(features)

        return x

    def forward(self, *args, return_embedding=False, return_label_features=False, **kwargs):
        if len(args) > 0:
            x = args[0]
        elif 'input_ids' in kwargs:
            x = kwargs['input_ids']
        elif 'x' in kwargs:
            x = kwargs['x']
        else:
            raise ValueError(f"No valid input found. args: {args}, kwargs: {kwargs.keys()}")

        return self._forward_impl(x, return_embedding=return_embedding,
                                  return_label_features=return_label_features)


def DINOv3_ViTB16(model_params):
    num_classes = model_params['num_classes']
    pretrained = model_params['pretrained']
    args = model_params['args']

    model_name = 'vit_base_patch16_dinov3.lvd1689m'

    model = DINOv3(model_name, pretrained, num_classes)
    model.num_features = 768

    if pretrained:
        print(f'Loaded DINOv3-ViT-B/16 pretrained weights from timm (lvd1689m)')

    # Add LoRA
    if args and hasattr(args, 'use_lora') and args.use_lora:
        print("Use LoRA")
        model = _apply_lora_to_model(model, args)
    else:
        _freeze_backbone_layers(model, args)

    return model
