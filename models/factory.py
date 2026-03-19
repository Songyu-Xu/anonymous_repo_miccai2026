import logging
import os
from urllib import request

import torch

from ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from models.tresnet import TResnetM, TResnetL, TResnetXL
from models.resnet import Resnet18, Resnet34, Resnet50, Resnet50_timm, Resnet101_timm
from models.surgenet import SurgeNetS18, SurgeNetS18_XL, SurgeNetS18_Small
from models.dinov3 import (DINOv3_ViTS16, DINOv3_ViTB16, DINOv3_ViTL16, DINOv3_ViTH16,
    DINOv3_ConvNeXtT, DINOv3_ConvNeXtS, DINOv3_ConvNeXtB, DINOv3_ConvNeXtL)

def create_model(args,load_head=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes, 'pretrained': args.pretrained}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'resnet_50_timm':
        model = Resnet50_timm(model_params)
    elif args.model_name == 'dinov3_vitb16':
        model = DINOv3_ViTB16(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    if args.use_ml_decoder:
        print("Use ML-Decoder")
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model
