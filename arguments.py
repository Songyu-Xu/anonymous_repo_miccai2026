import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set MLC Pipeline', add_help=False)

    # data conf.
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', default='/path/to/data')
    parser.add_argument('--image_size', type=int, default=448)

    # learning schedule
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)

    # schedule
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                         help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3457', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true', help='')

    # model
    parser.add_argument('--model-name', default='resnet_50_timm') #tresnet_m
    parser.add_argument('--model-path', default='')
    parser.add_argument('--num-classes', default=80, type=int)
    parser.add_argument('--pretrained', default=True, type=lambda x: (str(x).lower() == 'true'))

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)

    # save gpu memory
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')

    # device and log
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print_freq', default=200, type=int, help='number of iteration to print training logs')

    # noise setup
    parser.add_argument('--augment', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--noise_type', default='mislabeling', type=str)
    parser.add_argument('--noise_rate', default=0.0, type=float)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    # noisy label dir (pickle files)
    parser.add_argument('--noisy_label_dir', type=str, 
                    default='/home/songyu/Project/KAIST_BalanceMix/datasets/corrupted_labels/noisy_labels',
                    help='path to noisy label (pickle files) directory')

    # balancemix parameters
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--mixup_coef', default=4.0, type=float)
    parser.add_argument('--relabel_threshold', default=0.975, type=float)

    # Default BCE
    parser.add_argument('--use_default_bce', action='store_true',
                        help='Use default BCE training without minority sampler and mixup')

    # Freeze Backbone (DINOv3)
    parser.add_argument('--freeze_backbone_layers', default=None, type=int,
                        help='Number of layers freezed in backbone. -1 freeze all.')

    # LoRA
    parser.add_argument('--use-lora', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-rank', default=8, type=int,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', default=16.0, type=float,
                        help='LoRA alpha scaling factor (default: 16.0)')
    parser.add_argument('--lora-dropout', default=0.1, type=float,
                        help='LoRA dropout rate (default: 0.1)')
    parser.add_argument('--lora-target-modules', nargs='+', default=None,
                        help='Target modules for LoRA (e.g., qkv proj). If not specified, auto-select based on model type')

    # neighbor-smoothing
    parser.add_argument('--use-neighbor-smoothing', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Use neighbor-smoothing')
    
    # CSCC
    parser.add_argument('--use_cscc', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='use CSCC weighting')
    parser.add_argument('--cscc_k', default=20, type=int,
                        help='number of neighbors for CSCC')
    parser.add_argument('--cscc_gamma', default=0.5, type=float,
                        help='base weight for CSCC')
    parser.add_argument('--omega', default=0.333, type=float,
                       help='weight for CSCC loss in total loss: loss = omega * L_cscc + (1-omega) * L_balancemix')

    return parser
