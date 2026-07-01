import datetime
import os
import sys
import time
import argparse

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import random
from loss.loss import Loss
from torch.utils.data import Dataset, ConcatDataset


def _select_variant_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cafi_variant', default='lgce',
                        choices=['lgce', 'rmsin', 'CAFI-LGCE', 'CAFI-RMSIN'])
    known, _ = parser.parse_known_args()
    variant = known.cafi_variant.lower().replace('cafi-', '')
    root = os.path.dirname(os.path.abspath(__file__))
    variant_path = os.path.join(root, 'variants', variant)
    if variant == 'rmsin':
        sys.path.insert(0, os.path.join(variant_path, 'arc'))
    sys.path.insert(0, variant_path)
    return variant


CAFI_VARIANT = _select_variant_from_argv()
from lib import segmentation

class SwappedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        data = self.original_dataset[idx]
        image, target, l, l_att, nl, nl_att = data
        
        # 交换 l 和 nl，交换 l_att 和 nl_att，并取反 target
        new_data = (
            image,
            1 - target,  # 假设 target 是二分类标签（0/1）
            nl,          # 原 nl → 新 l
            nl_att,      # 原 nl_att → 新 l_att
            l,           # 原 l → 新 nl
            l_att       # 原 l_att → 新 nl_att
        )
        return new_data

def seed_everything(seed=333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


RRSISD_LARGE_TARGET_CATEGORIES = frozenset({
    'Expressway-Service-area',
    'airport',
    'baseballfield',
    'basketballcourt',
    'dam',
    'golffield',
    'groundtrackfield',
    'overpass',
    'stadium',
    'tenniscourt',
    'trainstation',
})

REFSEGRS_EXCLUDED_AUG_WORDS = ('van', 'bus', 'road marking')


def get_dataset_module(args, subset=False):
    dataset_name = getattr(args, 'rrsis_dataset', 'refsegrs')
    if dataset_name == 'rrsisd':
        module = 'data.bisubset_refer_bert' if subset else 'data.bidataset_refer_bert'
    else:
        module = 'data.binewsubset_refer_bert' if subset else 'data.binewdataset_refer_bert'
    return module


def get_dataset(image_set, transform, args):
    module = __import__(get_dataset_module(args, subset=False), fromlist=['ReferDataset'])
    ReferDataset = module.ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes

def get_subset(image_set, transform, args):
    module = __import__(get_dataset_module(args, subset=True), fromlist=['ReferDataset'])
    ReferDataset = module.ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def get_rrsisd_large_target_cat_ids(refer):
    return {
        cat_id for cat_id, name in refer.Cats.items()
        if name in RRSISD_LARGE_TARGET_CATEGORIES
    }


def has_short_negative_caption(dataset, idx, max_nega_tokens=20):
    return min(att.sum().item() for att in dataset.nega_attention_masks[idx]) <= max_nega_tokens


def refsegrs_sentence_is_allowed(dataset, idx):
    sentence = getattr(dataset, 'sentences', [''])[idx].lower()
    return not any(word in sentence for word in REFSEGRS_EXCLUDED_AUG_WORDS)


def is_valid_swap_aug_sample(dataset, idx, large_cat_ids=None, dataset_name=None, max_nega_tokens=20):
    if not has_short_negative_caption(dataset, idx, max_nega_tokens=max_nega_tokens):
        return False
    if dataset_name == 'rrsisd':
        ref_id = dataset.ref_ids[idx]
        cat_id = dataset.refer.Refs[ref_id]['category_id']
        if large_cat_ids is None:
            large_cat_ids = get_rrsisd_large_target_cat_ids(dataset.refer)
        return cat_id in large_cat_ids
    if dataset_name == 'refsegrs':
        return refsegrs_sentence_is_allowed(dataset, idx)
    return True


def build_swap_aug_indices(dataset, augset_size, dataset_name=None, max_nega_tokens=20):
    large_cat_ids = None
    if dataset_name == 'rrsisd':
        large_cat_ids = get_rrsisd_large_target_cat_ids(dataset.refer)
    valid_indices = [
        idx for idx in range(len(dataset))
        if is_valid_swap_aug_sample(
            dataset, idx,
            large_cat_ids=large_cat_ids,
            dataset_name=dataset_name,
            max_nega_tokens=max_nega_tokens,
        )
    ]
    random.shuffle(valid_indices)
    return valid_indices[:augset_size], valid_indices

# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def criterion(input, target):
    return Loss(weight=0.1)(input, target)

def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions , negative_sentences, negative_attentions = data
            image, target, sentences, attentions , negative_sentences, negative_attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True),\
                                                   negative_sentences.cuda(non_blocking=True),\
                                                   negative_attentions.cuda(non_blocking=True)


            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            negative_sentences = negative_sentences.squeeze(1)
            negative_attentions = negative_attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                negative_last_hidden_states = bert_model(negative_sentences, attention_mask=negative_attentions)[0]
                negative_embedding = negative_last_hidden_states.permute(0, 2, 1)
                negative_attentions =negative_attentions.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=attentions, nl_feats = negative_embedding, nl_mask = negative_attentions)
            else:
                output = model(image, sentences, l_mask=attentions, nl_feats = negative_sentences, nl_mask = negative_attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U

def front_door_train(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    #### freeze classifier
    # for param in model.module.classifier.parameters():
    #     param.requires_grad = False

    # # unfrozen modules in classifier
    # unfrozen_modules = [
    #     model.module.classifier.conv1_2,
    #     model.module.classifier.bn1_2,
    #     model.module.classifier.conv2_2,
    #     model.module.classifier.bn2_2,
    #     model.module.classifier.conv1_1,
    # ]

    # for module in unfrozen_modules:
    #     for param in module.parameters():
    #         param.requires_grad = True
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions , negative_sentences, negative_attentions = data
        image, target, sentences, attentions , negative_sentences, negative_attentions = image.cuda(non_blocking=True),\
                                                target.cuda(non_blocking=True),\
                                                sentences.cuda(non_blocking=True),\
                                                attentions.cuda(non_blocking=True),\
                                                negative_sentences.cuda(non_blocking=True),\
                                                negative_attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        negative_sentences = negative_sentences.squeeze(1)
        negative_attentions = negative_attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
            negative_last_hidden_states = bert_model(negative_sentences, attention_mask=negative_attentions)[0]
            negative_embedding = negative_last_hidden_states.permute(0, 2, 1)
            negative_attentions =negative_attentions.unsqueeze(dim=-1)
            output = model(image, embedding, l_mask=attentions, nl_feats = negative_embedding, nl_mask = negative_attentions)
        else:
            output = model(image, sentences, l_mask=attentions, nl_feats = negative_sentences, nl_mask = negative_attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    for param in model.module.backbone.parameters():
        param.requires_grad = True
    # for param in model.module.classifier.parameters():
    #     param.requires_grad = True
    #writer.add_scalar('Loss/train', train_loss, epoch)

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions , negative_sentences, negative_attentions = data
        image, target, sentences, attentions , negative_sentences, negative_attentions = image.cuda(non_blocking=True),\
                                                target.cuda(non_blocking=True),\
                                                sentences.cuda(non_blocking=True),\
                                                attentions.cuda(non_blocking=True),\
                                                negative_sentences.cuda(non_blocking=True),\
                                                negative_attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        negative_sentences = negative_sentences.squeeze(1)
        negative_attentions = negative_attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
            negative_last_hidden_states = bert_model(negative_sentences, attention_mask=negative_attentions)[0]
            negative_embedding = negative_last_hidden_states.permute(0, 2, 1)
            negative_attentions =negative_attentions.unsqueeze(dim=-1)
            output = model(image, embedding, l_mask=attentions, nl_feats = negative_embedding, nl_mask = negative_attentions)
        else:
            output = model(image, sentences, l_mask=attentions, nl_feats = negative_sentences, nl_mask = negative_attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    writer.add_scalar('Loss/train', train_loss, epoch)


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    subdataset, _ = get_subset("train",
                                get_transform(args=args),
                                args=args)
    # swapped augmentation subset setting
    augset_size = int(len(dataset) * args.swap_aug_ratio)
    augset_indices, valid_indices = build_swap_aug_indices(
        dataset,
        augset_size,
        dataset_name=args.rrsis_dataset,
        max_nega_tokens=args.swap_aug_max_nega_tokens,
    )
    if utils.get_rank() == 0:
        print(
            f"Swapped aug ({args.rrsis_dataset}): {len(valid_indices)} eligible "
            f"samples, using {len(augset_indices)} "
            f"({100 * len(augset_indices) / max(len(dataset), 1):.2f}%)"
        )
        if args.rrsis_dataset == 'rrsisd':
            print(f"  large-target categories: {sorted(RRSISD_LARGE_TARGET_CATEGORIES)}")
        if args.rrsis_dataset == 'refsegrs':
            print(f"  excluded RefSegRS words: {REFSEGRS_EXCLUDED_AUG_WORDS}")
    
    # 创建子集
    _augset = torch.utils.data.Subset(dataset, augset_indices)
    augset = SwappedDataset(_augset)
    merged_dataset = ConcatDataset([dataset, augset])
    
    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    sub_sampler = torch.utils.data.distributed.DistributedSampler(subdataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    # data loader
    data_loader = torch.utils.data.DataLoader(
        merged_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    data_loader_subset = torch.utils.data.DataLoader(
        subdataset, batch_size=args.batch_size,
        sampler=sub_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=False)
    # model initialization
    print(args.model)
    model = segmentation.__dict__['lavt_catt'](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
        params_to_optimize_fi = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
        params_to_optimize_fi = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    optimizer_fi = torch.optim.AdamW(params_to_optimize_fi,
                                  lr=0.25*args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler with warmup
    total_iters = len(data_loader) * args.epochs
    warmup_iters = int(total_iters * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        return (1 - (current_step - warmup_iters) / (total_iters - warmup_iters)) ** 0.9

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_iters_fi = len(data_loader_subset) * args.epochs
    warmup_iters_fi = int(total_iters_fi * 0.05)

    def lr_lambda_fi(current_step):
        if current_step < warmup_iters_fi:
            return float(current_step) / float(max(1, warmup_iters_fi))
        return (1 - (current_step - warmup_iters_fi) / (total_iters_fi - warmup_iters_fi)) ** 0.9

    lr_scheduler_fi = torch.optim.lr_scheduler.LambdaLR(optimizer_fi, lr_lambda_fi)
    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    #tensorboard
    writer = SummaryWriter('runs')

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        front_door_train(model, criterion, optimizer_fi, data_loader_subset, lr_scheduler_fi, epoch, args.print_freq,
                        iterations, bert_model)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model, writer)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)
        writer.add_scalar('IoU/train', iou, epoch)
        writer.add_scalar('overallIoU/train', overallIoU, epoch)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
    writer.close()
    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
