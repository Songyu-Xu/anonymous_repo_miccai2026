import os
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import resource
import utils.misc as utils
import argparse
from datasets import build_dataset
from arguments import get_args_parser
from models import create_model
from utils.helper_functions import ModelEma
from utils.scheduler import create_scheduler
from balancemix_engine import evaluate, evaluate_gmms, compute_neighbor_relationships, train_one_epoch_warmup, train_one_epoch_ssl, train_one_epoch_default, neighbor_label_smoothing, compute_cscc_weights
from loss_functions.losses_balancemix import BCELoss, RobustBCE, CSCCRobustBCE, SingleModalCSCCLoss, CombinedLoss
from torch.amp import GradScaler
from utils.statistics import ExampleLogger
import json, requests, traceback
from utils.helper_functions import add_weight_decay
import wandb
import datetime
import glob

def main(args):

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    device = torch.device(args.device)

    # Init WandB
    if utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if hasattr(args, 'eval') and args.eval:
            run_name = f"TEST_{args.dataset_file}_{timestamp}"
        else:
            method_tag = "bce" if (hasattr(args, 'use_default_bce') and args.use_default_bce) else ("cscc" if (hasattr(args, 'use_cscc') and args.use_cscc) else "balancemix")
            if hasattr(args, 'use_cscc') and args.use_cscc:
                run_name = f"omega{args.omega}_{args.dataset_file}_{method_tag}_{args.noise_type}_{args.noise_rate}_{args.model_name}_{timestamp}"
            else:
                run_name = f"{args.dataset_file}_{method_tag}_{args.noise_type}_{args.noise_rate}_{args.model_name}_{timestamp}"

        code_dir = os.path.abspath(".")
        print(f"Code directory (abs_path): {code_dir}")
        run = wandb.init(
            project="KAIST_BalanceMix",
            name=run_name,
            config=dict(vars(args)),
            settings=wandb.Settings(code_dir=code_dir),
            save_code=True,
            mode="online"
        )
        run.log_code(root=code_dir)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print("Model created!")

    criterion = RobustBCE()
    gmm_criterion = BCELoss()
    cscc_criterion = CSCCRobustBCE()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True,
                                                          broadcast_buffers=False)
        model_without_ddp = model.module
    ema_model = ModelEma(model_without_ddp, 0.9997)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # optimizer
    def build_optimizer(model, args):
        if args.dataset_file in ['deepfashion', 'mini-openimages'] \
            or 'tresnet' in args.model_name or 'resnet_101' in args.model_name:
            parameters = add_weight_decay(model, args.weight_decay)
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        return optimizer

    optimizer = build_optimizer(model_without_ddp, args)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.eval:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            if 'ema_model' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
            
        dataset_test = build_dataset(image_set='test', args=args)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, num_workers=args.num_workers)
        test_stats = evaluate(model, ema_model, gmm_criterion, data_loader_test, device, epoch=0)  # Use data_loader_test
        return

    # prepare the target dataset
    dataset_train = build_dataset(image_set='train', args=args)
    num_train_data = len(dataset_train)
    dataset_train_val = build_dataset(image_set='train_val', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print("# train:", len(dataset_train), ", # val", len(dataset_val))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    weights = np.array([1. / num_train_data for _ in range(num_train_data)]) # uniform at random (init)
    sampler_train_reweight = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_train_data, replacement=True)
    sampler_train_val = torch.utils.data.RandomSampler(dataset_train_val)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data samplers
    if args.distributed:
        sampler_train = DistributedSampler(sampler_train)
        sampler_train_val = DistributedSampler(sampler_train_val)
        sampler_val = DistributedSampler(sampler_val, shuffle=False)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    batch_sampler_train_reweight = torch.utils.data.BatchSampler(
        sampler_train_reweight, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)
    data_loader_train_reweight = DataLoader(dataset_train, batch_sampler=batch_sampler_train_reweight,
                                   num_workers=args.num_workers)
    data_loader_train_val = DataLoader(dataset_train_val, args.batch_size, sampler=sampler_train_val,
                                 drop_last=False, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)

    # training
    print("Start training")
    start_time = time.time()
    scaler = GradScaler('cuda')
    data_stats = {}
    best_ema_mAP = 0.0
    best_regular_mAP = 0.0
    best_ema_model = ema_model
    best_regular_model = model_without_ddp

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            if 'ema_model' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            args.start_epoch = checkpoint['epoch'] + 1

            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            sampler_train.set_epoch(epoch)

        if len(data_stats) != 0:
            acc_m_conf = None
            for key, value in data_stats.items():
                idx, example = key, value
                m_conf = example.get_conf_info()
                if acc_m_conf is None:
                    acc_m_conf = m_conf
                else:
                    acc_m_conf += m_conf
            m_conf = acc_m_conf / total_num

            m_conf = np.clip(np.array(m_conf), a_min=1e-5, a_max=0.99999)

            for key, value in data_stats.items():
                idx, example = key, value
                score = example.noisy_label * (1.0/m_conf[0, :]) + (1 - example.noisy_label) * (1.0/m_conf[1, :])
                score = np.clip(score, a_min=0.0, a_max=1e5)
                weights[idx] = np.sum(score)
            weights = np.nan_to_num(weights, nan=1e-8, posinf=1e5, neginf=1e-8)
            weights = np.maximum(weights, 1e-8)
            weights = weights / weights.sum()

            # update minority sampler
            sampler_train_reweight = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_train_data, replacement=True)
            batch_sampler_train_reweight = torch.utils.data.BatchSampler(
                sampler_train_reweight, args.batch_size, drop_last=True)
            data_loader_train_reweight = DataLoader(dataset_train, batch_sampler=batch_sampler_train_reweight,
                                                    num_workers=args.num_workers)

        if hasattr(args, 'use_default_bce') and args.use_default_bce:
            print("Train defualt BCE")
            train_stats, logs = train_one_epoch_default(
                model, ema_model, BCELoss(), data_loader_train, scaler, optimizer, device, epoch,
                print_freq=args.print_freq
            )
        elif epoch < args.warmup:
            print("Train warm-up")
            # warmup phase
            train_stats, logs = train_one_epoch_warmup(
                model, ema_model, criterion, data_loader_train, data_loader_train_reweight, scaler, optimizer, device, epoch,
                print_freq=args.print_freq
                )
        else:
            print("Train SSL")
            # 1. update GMMs for label selection
            # Estimate the clean probability of each sample using a two-component Gaussian Mixture Model (GMM)
            train_val_stats, idx_to_prob, idx_to_keep = \
               evaluate_gmms(model, gmm_criterion, data_loader_train_val, device, epoch, print_freq=args.print_freq)

            # 2. Compute CSCC weights
            if hasattr(args, 'use_cscc') and args.use_cscc:
                print("Computing CSCC weights")
                cscc_weights = compute_cscc_weights(
                    model=ema_model.module if hasattr(ema_model, 'module') else ema_model,
                    data_loader=data_loader_train,
                    device=device,
                    K=args.cscc_k,  # number of neighbors default 20
                    gamma=args.cscc_gamma  # base weight default 0.5
                )
            else:
                cscc_weights = None

            data_loader_train.dataset.setMultiView(True)
            data_loader_train_reweight.dataset.setMultiView(True)

            status_msg = "Train SSL with CSCC" if (
                        hasattr(args, 'use_cscc') and args.use_cscc) else "Train SSL"
            print(status_msg)

            train_stats, logs = train_one_epoch_ssl(
                model, ema_model, 
                criterion, cscc_criterion,
                data_loader_train, data_loader_train_reweight,
                scaler, optimizer, device, epoch,
                relabel_threshold=args.relabel_threshold,
                mixup_coef=args.mixup_coef,
                idx_to_prob=idx_to_prob,
                cscc_weights=cscc_weights,
                print_freq=args.print_freq,
                omega=args.omega
            )

        lr_scheduler.step(epoch)

        if len(data_stats) == 0:
            pos_total_num = np.sum(logs['noisy_label'], axis=0)
            neg_total_num = len(logs['index']) - pos_total_num
            total_num = np.stack([pos_total_num, neg_total_num])

        for sample_idx, _noisy_label, _pred in zip(logs['index'], logs['noisy_label'], logs['pred']):
            if sample_idx not in data_stats:
                data_stats[sample_idx] = ExampleLogger(sample_idx, _noisy_label)
            data_stats[sample_idx].add(_pred)

        test_stats = evaluate(model, ema_model, gmm_criterion, data_loader_val, device, epoch)
        if best_ema_mAP < test_stats['mAP_ema']:
            best_ema_model = ema_model
            best_ema_mAP = test_stats['mAP_ema']

            if args.output_dir and utils.is_main_process():
                old_best_checkpoints = glob.glob(str(output_dir / 'best_ema_epoch*.pth'))
                for old_ckpt in old_best_checkpoints:
                    os.remove(old_ckpt)
                    print(f"Removed old best checkpoint: {old_ckpt}")
                best_checkpoint_path = output_dir / f'best_ema_epoch{epoch:04d}_mAP{best_ema_mAP:.2f}.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'ema_model': best_ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'mAP': best_ema_mAP,
                    'args': args,
                }, best_checkpoint_path)
                print(f"Saved new best checkpoint for ema model: {best_checkpoint_path}")
        if best_regular_mAP < test_stats['mAP_regular']:
            best_regular_model = model_without_ddp
            best_regular_mAP = test_stats['mAP_regular']
            if args.output_dir and utils.is_main_process():
                old_best_checkpoints = glob.glob(str(output_dir / 'best_regular_epoch*.pth'))
                for old_ckpt in old_best_checkpoints:
                    os.remove(old_ckpt)
                    print(f"Removed old best checkpoint: {old_ckpt}")
                best_checkpoint_path = output_dir / f'best_regular_epoch{epoch:04d}_mAP{best_regular_mAP:.2f}.pth'
                utils.save_on_master({
                    'model': best_regular_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'mAP': best_regular_mAP,
                    'args': args,
                }, best_checkpoint_path)
                print(f"Saved new best checkpoint for regular model: {best_checkpoint_path}")


        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.warmup:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'best_ema_mAP': best_ema_mAP}

        if utils.is_main_process():
            training_phase = "warmup" if epoch < args.warmup else "ssl"

            log_dict = {
                "epoch": epoch,
                "training_phase": training_phase,
                "train_loss": train_stats.get('loss', 0),
                "train_loss_balancemix": train_stats.get('loss_balancemix', 0),
                "train_loss_cscc": train_stats.get('loss_cscc', 0),
                "train_lr": train_stats.get('lr', 0),
                "test_loss": test_stats.get('loss', 0),
                "test_mAP_regular": test_stats.get('mAP_regular', 0),
                "test_mAP_ema": test_stats.get('mAP_ema', 0),
                "test_mAP_best": test_stats.get('mAP_best', 0),
                "best_ema_mAP": best_ema_mAP,
                "best_regular_mAP": best_regular_mAP
            }

            if 'APs_regular' in test_stats:
                for class_name, ap_value in test_stats['APs_regular'].items():
                    log_dict[f"test_AP/{class_name}"] = ap_value

            wandb.log(log_dict, step=epoch)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        wandb.finish()
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MLC training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir == '':
        dataset_file = args.dataset_file
        backbone = args.model_name

        method_tag = "bce" if (hasattr(args, 'use_default_bce') and args.use_default_bce) else "balancemix"
        noise_rate_str = str(int(args.noise_rate * 100)) if args.noise_rate > 0 else "0"
        args.output_dir = f"{dataset_file}_{method_tag}_{args.noise_type}{noise_rate_str}_{backbone}_warmup{args.warmup}_epochs{args.epochs}_omega{args.omega}"
        args.output_dir = os.path.join('logs', args.output_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('log', args.output_dir)

    if args.dataset_file == 'voc':
        from datasets.voc.utils import create_data_lists
        create_data_lists(
            voc07_path=os.path.join(args.data_path, 'VOC2007'),
            voc12_path=os.path.join(args.data_path, 'VOC2012'),
            output_folder='.')
    main(args)