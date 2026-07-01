import datetime
import os
import sys
import time
import argparse

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
from visualization import save_images
from visualization import save_masks
from visualization import save_sentences, load_vocab
from visualization import view_attention_map


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


def get_dataset(image_set, transform, args):
    if getattr(args, 'rrsis_dataset', 'refsegrs') == 'rrsisd':
        from data.bidataset_refer_bert import ReferDataset
    else:
        from data.binewdataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    batch_i=0
    vocab = load_vocab('./pretrained_weights/bert/vocab.txt')
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, negative_sentences, negative_attentions = data
            image, target, sentences, attentions, negative_sentences, negative_attentions = image.to(device), target.to(device), \
                                                                                            sentences.to(device), attentions.to(device), \
                                                                                            negative_sentences.to(device), negative_attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            negative_sentences = negative_sentences.squeeze(1)
            negative_attentions = negative_attentions.squeeze(1)
            target = target.cpu().data.numpy()
            #保存图片
            batch_i += 1
            #print((image.cpu().shape))           
            #save_images(image.cpu()[0],f"./visualization/image{batch_i}.PNG")
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    negative_last_hidden_states = bert_model(negative_sentences[:, :, j], attention_mask=negative_attentions[:, :, j])[0]
                    negative_embedding = negative_last_hidden_states.permute(0, 2, 1)
                    #negative_attentions =negative_attentions.unsqueeze(dim=-1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1), nl_feats = negative_embedding, nl_mask = negative_attentions[:, :, j].unsqueeze(-1))
                    #features = model.backbone(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    #features = model.backbone(image, sentences[:, :, j], l_mask=attentions[:, :, j])                  
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j], nl_feats=negative_sentences[:, :, j], nl_mask=negative_attentions[:, :, j])
                #############---------------------#################
                ##########attention_visualization##################
                #features = features.cpu()
                #view_attention_map(features, image.cpu(), f"image{batch_i}_attention{j}")
                #############---------------------#################
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                #保存可视化结果：
                #save_sentences(sentences[:, :, j].cpu(), vocab ,'./visualization/refer.txt', batch_i)
                #print(output.shape)
                #for i in range(0,len(output[0])):
                #save_masks(image.cpu(),output,f"image{batch_i}_mask{j}")
                #save_images(output_mask,target)

                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask, negative_sentences, negative_attentions
            if bert_model is not None:
                del last_hidden_states, embedding, negative_last_hidden_states, negative_embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__['lavt_catt'](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
