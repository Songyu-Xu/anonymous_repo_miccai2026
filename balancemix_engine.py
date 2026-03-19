import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from utils.helper_functions import mAP, get_clean_prob_with_gmms
from torch.amp import autocast
import utils.misc as misc
import numpy as np
import faiss
import torch.nn.functional as F
from sklearn.decomposition import PCA

def train_one_epoch_warmup(model, ema_model, criterion, data_loader, weighted_data_loader, scaler, optimizer,
                    device, epoch, mixup_coef=1.0, print_freq=100):

    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq
    preds_regular = []
    targets = []

    local_indexes, local_gt_labels, local_noisy_labels, local_preds = None, None, None, None

    balanced_sampler = iter(weighted_data_loader)
    for index, sample, target, raw_target in metric_logger.log_every(data_loader, print_freq, header):
        b_index, b_sample, b_target, b_raw_target = next(balanced_sampler)

        index, sample, target = index.to(device), sample.to(device), target.to(device)
        b_index, b_sample, b_target = b_index.to(device), b_sample.to(device), b_target.to(device)
        gt_target = raw_target.to(device)

        merged_index = torch.cat([index, b_index], dim=0)
        merged_sample = torch.cat([sample, b_sample], dim=0)
        merged_target = torch.cat([target, b_target], dim=0)

        col_ind = torch.randperm(merged_index.size(0))
        b_merged_index, b_merged_sample, b_merged_target = \
            merged_index[col_ind], merged_sample[col_ind], merged_target[col_ind]

        l = np.random.beta(mixup_coef, mixup_coef)
        mixed_input = l * merged_sample + (1 - l) * b_merged_sample
        mixed_target = l * merged_target + (1 - l) * b_merged_target

        with autocast('cuda', enabled=True):
            mixed_output = model(mixed_input).float()
            loss = criterion(mixed_output, mixed_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        model_without_ddp = model.module if hasattr(model, 'module') else model
        ema_model.update(model_without_ddp)

        reduced_loss = misc.reduce_dict({"loss": loss})
        reduced_loss = sum(reduced_loss.values())
        loss_value = reduced_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        with autocast('cuda', enabled=True):
            with torch.no_grad():
                output = model(sample).float()
        preds_regular.append(torch.sigmoid(output).cpu().detach())
        targets.append(target.cpu().detach())

        index = index.detach()
        if local_indexes is None:
            local_indexes = index
            local_preds = torch.sigmoid(output)
            local_noisy_labels = target
            local_gt_labels = gt_target
        else:
            local_indexes = torch.cat([local_indexes, index], dim=0)
            local_noisy_labels = torch.cat([local_noisy_labels, target], dim=0)
            local_gt_labels = torch.cat([local_gt_labels, gt_target], dim=0)
            local_preds = torch.cat([local_preds, torch.sigmoid(output)], dim=0)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    reduced_mAP = misc.reduce_dict({"mAP": torch.as_tensor(mAP_score_regular).to(device)})
    mAP_regular_value = reduced_mAP["mAP"].item()
    stats["mAP"] = mAP_regular_value
    APs_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(), return_all=True)
    reduced_APs = misc.reduce_dict({k: torch.as_tensor(v).to(device) for k, v in APs_score_regular.items()})
    reduced_APs = {k: v.item() for k, v in reduced_APs.items()}
    print('Entire AP: ', reduced_APs)

    merged_indexes = [item.cpu() for item in misc.all_gather(local_indexes)]
    merged_noisy_labels = [item.cpu() for item in misc.all_gather(local_noisy_labels)]
    merged_gt_labels = [item.cpu() for item in misc.all_gather(local_gt_labels)]
    merged_preds = [item.detach().cpu() for item in misc.all_gather(local_preds)]

    merged_indexes = torch.cat(merged_indexes, dim=0)
    merged_noisy_labels = torch.cat(merged_noisy_labels, dim=0)
    merged_gt_labels = torch.cat(merged_gt_labels, dim=0)
    merged_preds = torch.cat(merged_preds, dim=0)

    logs = {
            'index': merged_indexes.numpy(),
            'gt_label': merged_gt_labels.numpy(),
            'noisy_label': merged_noisy_labels.numpy(),
            'pred': merged_preds.numpy(),
            }

    ids = np.argsort(logs['index'])
    logs = {k: v[ids] for k, v in logs.items()}

    print("Averaged stats:", stats)
    return stats, logs


def train_one_epoch_ssl(model, ema_model, criterion_balancemix, criterion_cscc, data_loader, weighted_data_loader, scaler, optimizer,
                    device, epoch, print_freq=100,
                    relabel_threshold=0.95,
                    idx_to_prob=None,
                    cscc_weights=None,
                    mixup_coef=4.0,
                    omega=None):

    model.train()
    criterion_balancemix.train()
    criterion_cscc.train()

    metric_logger = misc.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_balancemix', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cscc', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    preds_regular = []
    targets = []

    # index
    local_indexes, local_gt_labels, local_noisy_labels, local_preds = None, None, None, None

    if idx_to_prob is not None:
        idx_to_prob = idx_to_prob.to(device)
    # Preparing dual data stream
    balanced_sampler = iter(weighted_data_loader)
    for index, sample1, sample2, target, raw_target in metric_logger.log_every(data_loader, print_freq, header):
        b_index, b_sample1, b_sample2, b_target, b_raw_target = next(balanced_sampler)

        batch_size = index.shape[0]
        index, sample1, sample2, target = index.to(device), sample1.to(device), sample2.to(device), target.to(device)
        b_index, b_sample1, b_sample2, b_target = b_index.to(device), b_sample1.to(device), b_sample2.to(device), b_target.to(device)
        gt_target = raw_target.to(device)

        # Step 1: label processing to get relabeled labels
        sample, relabel_target, total_mask, relabel_mask, clean_prob, pred_log, target_log =\
            label_processing(model, index, sample1, sample2, target, idx_to_prob, sharpening=False, relabel_threshold=relabel_threshold,
                             noise_type=data_loader.dataset.noise_type, noise_rate=data_loader.dataset.noise_rate)
        b_sample, b_relabel_target, b_total_mask, b_relabel_mask, b_clean_prob, b_pred_log, b_target_log =\
            label_processing(model, b_index, b_sample1, b_sample2, b_target, idx_to_prob, sharpening=False, relabel_threshold=relabel_threshold,
                             noise_type=data_loader.dataset.noise_type, noise_rate=data_loader.dataset.noise_rate)

        if cscc_weights is not None:
            batch_cscc_weights = cscc_weights[index]
            b_batch_cscc_weights = cscc_weights[b_index]
        else: # if no cscc_weights, set all to 1.0
            batch_cscc_weights = torch.ones(batch_size, device=device)
            b_batch_cscc_weights = torch.ones(b_index.shape[0], device=device)

        batch_cscc_weights = torch.cat([batch_cscc_weights, batch_cscc_weights], dim=0)
        b_batch_cscc_weights = torch.cat([b_batch_cscc_weights, b_batch_cscc_weights], dim=0)

        # Step 3 mixing
        col_ind = torch.randperm(b_sample.size(0))
        b_sample, b_relabel_target = b_sample[col_ind], b_relabel_target[col_ind]
        b_target_log = b_target_log[col_ind]
        b_batch_cscc_weights = b_batch_cscc_weights[col_ind]

        l = np.random.beta(mixup_coef, mixup_coef)
        l = max(l, 1.0 - l)
        mixed_input = l * sample + (1 - l) * b_sample
        mixed_target = l * relabel_target + (1 - l) * b_relabel_target
        mixed_noisy_target = l * target_log + (1 - l) * b_target_log
        mixed_clean_prob = clean_prob
        mixed_cscc_weight = l * batch_cscc_weights + (1 - l) * b_batch_cscc_weights
        mixed_mask = total_mask

        with autocast('cuda', enabled=True):
            mixed_output = model(mixed_input).float()
            # Loss 1: CSCC loss - using mixed_noisy_target
            loss_cscc = criterion_cscc(mixed_output, mixed_noisy_target,
                                      cscc_weight=mixed_cscc_weight)
            # Loss 2: BalanceMix loss - using relabeled mixed_target
            loss_balancemix = criterion_balancemix(mixed_output, mixed_target, mask=mixed_mask, 
                             clean_prob=mixed_clean_prob)
            
            # Total Loss:
            # Total loss ver1
            # alpha = 0.5  # weight for cscc loss
            # loss = loss_balancemix + alpha * loss_cscc

            # Total loss ver2
            # omega = 0.333
            loss = omega * loss_cscc + (1-omega) * loss_balancemix  # omega passed from args.omega

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # ema update
        model_without_ddp = model.module if hasattr(model, 'module') else model
        ema_model.update(model_without_ddp)

        # reduce losses
        reduced_loss = misc.reduce_dict({"loss": loss,
                                         "loss_balancemix": loss_balancemix,
                                         "loss_cscc": loss_cscc})
        reduced_loss = sum(reduced_loss.values())
        loss_value = reduced_loss.item()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_balancemix=loss_balancemix)
        metric_logger.update(loss_cscc=loss_cscc)

        # compute mAP
        preds_regular.append(pred_log.cpu().detach())
        targets.append(target_log.cpu().detach())

        # log aggregate ###
        index = index.detach()
        if local_indexes is None:
            local_indexes = index
            local_preds = pred_log[:batch_size, :]
            local_noisy_labels = target
            local_gt_labels = gt_target
        else:
            local_indexes = torch.cat([local_indexes, index], dim=0)
            local_noisy_labels = torch.cat([local_noisy_labels, target], dim=0)
            local_gt_labels = torch.cat([local_gt_labels, gt_target], dim=0)
            local_preds = torch.cat([local_preds, pred_log[:batch_size, :]], dim=0)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    reduced_mAP = misc.reduce_dict({"mAP": torch.as_tensor(mAP_score_regular).to(device)})
    mAP_regular_value = reduced_mAP["mAP"].item()
    stats["mAP"] = mAP_regular_value

    merged_indexes = [item.cpu() for item in misc.all_gather(local_indexes)]
    merged_noisy_labels = [item.cpu() for item in misc.all_gather(local_noisy_labels)]
    merged_gt_labels = [item.cpu() for item in misc.all_gather(local_gt_labels)]
    merged_preds = [item.detach().cpu() for item in misc.all_gather(local_preds)]

    merged_indexes = torch.cat(merged_indexes, dim=0)
    merged_noisy_labels = torch.cat(merged_noisy_labels, dim=0)
    merged_gt_labels = torch.cat(merged_gt_labels, dim=0)
    merged_preds = torch.cat(merged_preds, dim=0)

    logs = {
            'index': merged_indexes.numpy(),
            'gt_label': merged_gt_labels.numpy(),
            'noisy_label': merged_noisy_labels.numpy(),
            'pred': merged_preds.numpy(),
            }

    ids = np.argsort(logs['index'])
    logs = {k: v[ids] for k, v in logs.items()}

    print("Averaged stats:", stats)

    return stats, logs


def train_one_epoch_default(model, ema_model, criterion, data_loader, scaler, optimizer,
                            device, epoch, print_freq=100):
    """
    Default BCE training: no minority sampler, no mixup
    Just standard training with BCE loss
    """
    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq

    preds_regular = []
    targets = []

    local_indexes, local_gt_labels, local_noisy_labels, local_preds = None, None, None, None

    for index, sample, target, raw_target in metric_logger.log_every(data_loader, print_freq, header):
        index, sample, target = index.to(device), sample.to(device), target.to(device)
        gt_target = raw_target.to(device)

        # forward pass - no mixup, just original samples
        with autocast('cuda', enabled=True):
            output = model(sample).float()
            loss = criterion(output, target, merge=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        model_without_ddp = model.module if hasattr(model, 'module') else model
        ema_model.update(model_without_ddp)

        reduced_loss = misc.reduce_dict({"loss": loss})
        reduced_loss = sum(reduced_loss.values())
        loss_value = reduced_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        preds_regular.append(torch.sigmoid(output).cpu().detach())
        targets.append(target.cpu().detach())

        index = index.detach()
        if local_indexes is None:
            local_indexes = index
            local_preds = torch.sigmoid(output)
            local_noisy_labels = target
            local_gt_labels = gt_target
        else:
            local_indexes = torch.cat([local_indexes, index], dim=0)
            local_noisy_labels = torch.cat([local_noisy_labels, target], dim=0)
            local_gt_labels = torch.cat([local_gt_labels, gt_target], dim=0)
            local_preds = torch.cat([local_preds, torch.sigmoid(output)], dim=0)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    reduced_mAP = misc.reduce_dict({"mAP": torch.as_tensor(mAP_score_regular).to(device)})
    mAP_regular_value = reduced_mAP["mAP"].item()
    stats["mAP"] = mAP_regular_value

    APs_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(), return_all=True)
    reduced_APs = misc.reduce_dict({k: torch.as_tensor(v).to(device) for k, v in APs_score_regular.items()})
    reduced_APs = {k: v.item() for k, v in reduced_APs.items()}
    print('Entire AP: ', reduced_APs)

    merged_indexes = [item.cpu() for item in misc.all_gather(local_indexes)]
    merged_noisy_labels = [item.cpu() for item in misc.all_gather(local_noisy_labels)]
    merged_gt_labels = [item.cpu() for item in misc.all_gather(local_gt_labels)]
    merged_preds = [item.detach().cpu() for item in misc.all_gather(local_preds)]

    merged_indexes = torch.cat(merged_indexes, dim=0)
    merged_noisy_labels = torch.cat(merged_noisy_labels, dim=0)
    merged_gt_labels = torch.cat(merged_gt_labels, dim=0)
    merged_preds = torch.cat(merged_preds, dim=0)

    logs = {
        'index': merged_indexes.numpy(),
        'gt_label': merged_gt_labels.numpy(),
        'noisy_label': merged_noisy_labels.numpy(),
        'pred': merged_preds.numpy(),
    }

    ids = np.argsort(logs['index'])
    logs = {k: v[ids] for k, v in logs.items()}

    print("Averaged stats:", stats)
    return stats, logs


def label_processing(model, index, sample1, sample2, target, idx_to_prob,
                     soft_labels=None,
                     relabel_threshold=0.95,
                     sharpening=False, noise_type='unknown', noise_rate=1.0):
    sample = torch.cat([sample1, sample2], dim=0).contiguous()

    _size = sample1.shape[0]

    if soft_labels is not None:
        batch_soft_labels = torch.index_select(soft_labels, dim=0, index=index)
        x_pred = batch_soft_labels
    else:
        with autocast('cuda', enabled=True):
            with torch.no_grad():
                output = model(sample).float()
        pred = torch.sigmoid(output)
        x_pred = (pred[:_size, :] + pred[_size:, :]) / 2.0

    # identified clean mask.
    clean_prob = torch.index_select(idx_to_prob, dim=0, index=index)
    clean_mask = torch.gt(clean_prob, 0.5)

    if noise_type in ['single', 'missing'] or noise_rate == 0.0:
        # mask addition: pos label is absolutely correct.
        known_gt_mask = torch.gt(target, 0.5)
        clean_mask = clean_mask + known_gt_mask

    noise_mask = ~clean_mask

    if sharpening:
        y_pos = x_pred ** (1 / 0.5)
        y_neg = (1 - x_pred) ** (1 / 0.5)
        x_pred = y_pos / (y_neg + y_pos)

    relabel_pos_mask = torch.gt(x_pred, relabel_threshold) * noise_mask
    relabel_neg_mask = torch.lt(x_pred, 1 - relabel_threshold) * noise_mask
    relabel_mask = relabel_pos_mask + relabel_neg_mask
    clean_mask = clean_mask
    relabel_mask = relabel_mask
    total_mask = clean_mask + relabel_mask

    # relabeling
    relabel_y = ~relabel_mask * target + relabel_pos_mask * 1.0 + relabel_neg_mask * 0.0
    relabel_y = torch.cat([relabel_y, relabel_y], dim=0).detach()

    total_mask = torch.cat([total_mask, total_mask], dim=0)
    target = torch.cat([target, target], dim=0)
    clean_prob = torch.cat([clean_prob, clean_prob], dim=0)

    return sample, relabel_y, total_mask, relabel_mask, clean_prob, pred, target

@torch.no_grad()
def evaluate_gmms(ema_model, criterion, data_loader, device, epoch, print_freq=100):

    ema_model.eval()
    criterion.eval()
    metric_logger = misc.MetricLogger(delimiter=", ")
    header = 'GMMs: [{}]'.format(epoch)

    # index
    local_indexes, local_losses, local_clean_flags, local_noisy_labels, local_gt_labels = None, None, None, None, None
    for index, sample, target, raw_target in metric_logger.log_every(data_loader, print_freq, header):
        index, sample, target = index.to(device), sample.to(device), target.to(device)
        gt_target = raw_target.to(device)

        with autocast('cuda', enabled=True):
            output_ema = ema_model.module(sample).float()
            losses = criterion(output_ema, target, merge=False)
            index = index.detach()
            losses = torch.log(losses.detach().clamp(min=1e-8))
            clean_flags = torch.eq(gt_target, target).detach()

            if local_indexes is None:
                local_indexes, local_losses, local_clean_flags, local_noisy_labels, local_gt_labels = \
                    index, losses, clean_flags, target, gt_target
            else:
                local_indexes = torch.cat([local_indexes, index], dim=0)
                local_losses = torch.cat([local_losses, losses], dim=0)
                local_clean_flags = torch.cat([local_clean_flags, clean_flags], dim=0)
                local_noisy_labels = torch.cat([local_noisy_labels, target], dim=0)
                local_gt_labels = torch.cat([local_gt_labels, gt_target], dim=0)

    merged_indexes = [item.cpu() for item in misc.all_gather(local_indexes)]
    merged_losses = [item.cpu() for item in misc.all_gather(local_losses)]
    merged_clean_flags = [item.cpu() for item in misc.all_gather(local_clean_flags)]
    merged_noisy_labels = [item.cpu() for item in misc.all_gather(local_noisy_labels)]
    merged_gt_labels = [item.cpu() for item in misc.all_gather(local_gt_labels)]
    merged_indexes = torch.cat(merged_indexes, dim=0)
    merged_losses = torch.cat(merged_losses, dim=0)
    merged_clean_flags = torch.cat(merged_clean_flags, dim=0)
    merged_noisy_labels = torch.cat(merged_noisy_labels, dim=0)
    merged_gt_labels = torch.cat(merged_gt_labels, dim=0)
    merged_indexes = merged_indexes.numpy()

    data_size, num_classes = merged_losses.shape

    # index -> prob
    idx_to_prob = {}
    idx_to_keep = {}
    check_duplicate = set()
    # 1. for each class, build gmms
    for class_id in range(num_classes):
        # idx
        ids_per_class = merged_indexes
        # loss
        losses_per_class = merged_losses[:, class_id].numpy()
        # noisy label
        labels_per_class = merged_noisy_labels[:, class_id].numpy()
        # whether to clean or noise
        flags_per_class = merged_clean_flags[:, class_id].numpy()
        # split losses into two subsets for pos and neg (w.r.t noisy labels)
        pos_ids, pos_flags, pos_losses = [], [], []
        neg_ids, neg_flags, neg_losses = [], [], []
        for idx, loss in enumerate(losses_per_class):
            id = ids_per_class[idx]
            if id not in check_duplicate:
                check_duplicate.add(id)
            else:
                continue

            flag = flags_per_class[idx]
            label = labels_per_class[idx]
            if label == 1:
                pos_ids.append(id)
                pos_flags.append(flag)
                pos_losses.append(loss)
            else:
                neg_ids.append(id)
                neg_flags.append(flag)
                neg_losses.append(loss)

        # clear duplication set
        check_duplicate.clear()
        pos_prob = get_clean_prob_with_gmms(pos_losses)
        neg_prob = get_clean_prob_with_gmms(neg_losses)

        # put clean probas -> [ids, # classes]
        for id, prob, flag in zip(pos_ids, pos_prob, pos_flags):
            if id not in idx_to_prob:
                idx_to_prob[id] = []
                idx_to_keep[id] = []
            idx_to_prob[id].append(prob)

            if flag:
                idx_to_keep[id].append(1.0)
            else:
                idx_to_keep[id].append(0.0)

        for id, prob, flag in zip(neg_ids, neg_prob, neg_flags):
            if id not in idx_to_prob:
                idx_to_prob[id] = []
                idx_to_keep[id] = []
            idx_to_prob[id].append(prob)

            if flag:
                idx_to_keep[id].append(1.0)
            else:
                idx_to_keep[id].append(0.0)

    idx_to_prob = dict(sorted(idx_to_prob.items()))
    idx_to_prob = torch.tensor([*idx_to_prob.values()]).to(device).detach()
    idx_to_keep = dict(sorted(idx_to_keep.items()))
    idx_to_keep = torch.tensor([*idx_to_keep.values()]).to(device).detach()
    idx_to_prob = idx_to_prob.cpu()

    # GMM evaluation
    merged_gt_labels = merged_gt_labels.numpy()
    merged_noisy_labels = merged_noisy_labels.numpy()
    temp_dic = {}
    for id, gt, noise in zip(merged_indexes, merged_gt_labels, merged_noisy_labels):
        temp_dic[id] = (gt, noise)
    _id, _gt, _noise = [], [], []
    for key, value in temp_dic.items():
        __gt, __noise = value
        _id.append(key)
        _gt.append(__gt)
        _noise.append(__noise)
    _id = np.array(_id)
    sorted_idx = np.argsort(_id)
    # Optimize
    merged_gt_labels = torch.from_numpy(np.array(_gt))[sorted_idx]
    merged_noisy_labels = torch.from_numpy(np.array(_noise))[sorted_idx]
    clean_masks = torch.gt(idx_to_prob, 0.5).cpu()

    pos_masks = torch.gt(merged_noisy_labels, 0.5)
    pos_num_clean = torch.count_nonzero(clean_masks * pos_masks, dim=0)
    neg_num_clean = torch.count_nonzero(clean_masks * ~pos_masks, dim=0)
    pos_num_total = torch.count_nonzero(pos_masks, dim=0)
    neg_num_total = torch.count_nonzero(~pos_masks, dim=0)
    min_pos_threshold = 0.2
    min_neg_threshold = 0.2
    pos_keep = ~torch.gt(pos_num_clean / pos_num_total, min_pos_threshold) * pos_masks
    neg_keep = ~torch.gt(neg_num_clean / neg_num_total, min_neg_threshold) * ~pos_masks
    keep = pos_keep + neg_keep
    idx_to_prob = 1.0 * keep + ~keep * idx_to_prob

    return None, idx_to_prob, idx_to_keep

@torch.no_grad()
def compute_cscc_weights(model, data_loader, device, K=20, gamma=0.5):
    """
    Unimodal CSCC weight computation
    """
    print(f"Computing CSCC weights (K={K}, gamma={gamma})...")
    
    # ========== 1. collect features and labels for all samples ==========
    all_features = []
    all_labels = []
    all_indices = []
    
    model.eval()
    original_multiview = data_loader.dataset.multi_view
    data_loader.dataset.setMultiView(False)
    
    with torch.no_grad():
        for batch_data in data_loader:
            indices = batch_data[0]
            images = batch_data[1].to(device)  # sample1
            labels = batch_data[3].to(device)  # target (noisy labels)
            
            # Extract features
            if hasattr(model, 'module'):
                features = model.module(images, return_embedding=True)
            else:
                features = model(images, return_embedding=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_indices.append(indices.cpu())
    
    data_loader.dataset.setMultiView(original_multiview)
    
    # ========== 2. Concatenate ==========
    all_indices = torch.cat(all_indices)
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    # ========== 3. Get dataset size ==========
    max_index = all_indices.max().item()
    dataset_size = max(max_index + 1, len(data_loader.dataset))
    # ========== 4. Sort by index ==========
    sorted_idx = torch.argsort(all_indices)
    sorted_indices = all_indices[sorted_idx]
    features = all_features[sorted_idx].to(device)
    labels = all_labels[sorted_idx].to(device)
    N = features.shape[0]
    
    # ========== 5. Compute similarity matrix ==========
    features_norm = F.normalize(features, dim=1)
    sim_matrix = torch.mm(features_norm, features_norm.t())
    sim_matrix.fill_diagonal_(0)
    # ========== 6. Find top-K neighbors ==========
    top_sim, nearest_indices = torch.topk(sim_matrix, k=K, dim=1)
    # ========== 7. Normalize similarity weights ==========
    top_sim_norm = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)
    # ========== 8. Compute soft labels (Eq. 1) ==========
    nearest_labels = labels[nearest_indices]
    weighted_labels = nearest_labels * top_sim_norm.unsqueeze(-1)
    soft_labels = weighted_labels.sum(dim=1)
    # ========== 9. Compute confidence weights (Eq. 2) ==========
    label_cosine_sim = F.cosine_similarity(labels, soft_labels, dim=1)
    weights = gamma + (1 - gamma) * label_cosine_sim

    weights_tensor = torch.ones(dataset_size, device=device)
    weights_tensor[sorted_indices] = weights
    
    # ========== 11. Print statistics ==========
    # print(f"CSCC weights - min: {weights.min():.3f}, max: {weights.max():.3f}, "
    #       f"mean: {weights.mean():.3f}, std: {weights.std():.3f}")
    # print(f"Weights tensor shape: {weights_tensor.shape}, Dataset size: {dataset_size}")
    
    model.train()
    return weights_tensor.detach()

@torch.no_grad()
def evaluate(model, ema_model, criterion, data_loader, device, epoch):
    print("Evaluation")

    model.eval()
    criterion.eval()

    Sig = torch.nn.Sigmoid()
    metric_logger = misc.MetricLogger(delimiter=", ")
    header = 'Epoch: [{}]'.format(epoch)
    local_pred, local_ema_pred, local_target = None, None, None

    for index, sample, target, raw_target in metric_logger.log_every(data_loader, 30, header):
        index = index.to(device)
        sample = sample.to(device)
        target = target.to(device)

        with autocast('cuda', enabled=True):
            output_regular = model(sample).float()
            output_ema = ema_model.module(sample).float()
            loss_regular = criterion(output_regular, target)
        reduced_loss_regular = misc.reduce_dict({"loss": loss_regular})
        loss_value_regular = reduced_loss_regular["loss"].item()
        metric_logger.update(loss=loss_value_regular)

        # compute mAP
        if local_pred is None:
            local_pred = Sig(output_regular).detach()
            local_ema_pred = Sig(output_ema).detach()
            local_target = target
        else:
            local_pred = torch.cat([local_pred, Sig(output_regular).detach()], dim=0)
            local_ema_pred = torch.cat([local_ema_pred, Sig(output_ema).detach()], dim=0)
            local_target = torch.cat([local_target, target], dim=0)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    merged_preds = [item.detach().cpu() for item in misc.all_gather(local_pred)]
    merged_preds = torch.cat(merged_preds, dim=0)
    merged_ema_preds = [item.detach().cpu() for item in misc.all_gather(local_ema_pred)]
    merged_ema_preds = torch.cat(merged_ema_preds, dim=0)
    merged_targets = [item.detach().cpu() for item in misc.all_gather(local_target)]
    merged_targets = torch.cat(merged_targets, dim=0)

    mAP_regular_value = mAP(merged_targets.numpy(), merged_preds.numpy())
    mAP_ema_value = mAP(merged_targets.numpy(), merged_ema_preds.numpy())
    mAP_best_value = max(mAP_regular_value, mAP_ema_value)

    APs_regular_value = mAP(merged_targets.numpy(), merged_preds.numpy(), return_all=True)
    print('Entire AP: ', APs_regular_value)

    stats["mAP_regular"] = mAP_regular_value
    stats["mAP_ema"] = mAP_ema_value
    stats["mAP_best"] = mAP_best_value
    stats["APs_regular"] = APs_regular_value

    print("Averaged stats:", stats)
    return stats