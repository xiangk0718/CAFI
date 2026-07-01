"""
Quantitative interpretability evaluation for CAFI (`lavt_catt`).

Key differences from baseline `eval_interpretability.py`:
- dataset: data.binewdataset_refer_bert (positive + negative text)
- model forward: model(image, pos_emb, l_mask, nl_feats, nl_mask)
- attention extraction: counterfactual PWAM forward (positive/negative maps)
"""

import datetime
import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from bert.modeling_bert import BertModel
import transforms as T
import utils

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

from eval_interpretability_base import (
    _aggregate,
    _infer_hw_from_count,
    attention_iou,
    compare_reports,
    compute_iou,
    deletion_insertion_auc,
    energy_inside_ratio,
    multiscale_consistency,
    pointing_game,
    print_report,
    saliency_decoder_l2,
    saliency_logits_fg,
    spatial_entropy,
    token_relevance_score,
)


class CAFIAttentionExtractor:
    """Monkey-patch CAFI SpatialImageLanguageAttention.forward and store maps."""

    def __init__(self, model):
        self.pos_attn_maps = {}
        self.neg_attn_maps = {}
        self.base_gate_feats = {}
        self.attn_maps = self.pos_attn_maps
        self._n_stages = len(model.backbone.layers)
        for i, layer in enumerate(model.backbone.layers):
            self._patch(layer.fusion.image_lang_att, i)

    def _patch(self, module, stage_idx):
        extractor = self

        def forward(x, l, l_mask, nl, nl_mask):
            B, HW = x.size(0), x.size(1)
            x = x.permute(0, 2, 1)
            l_mask = l_mask.permute(0, 2, 1)
            nl_mask = nl_mask.permute(0, 2, 1)

            query = module.f_query(x).permute(0, 2, 1)

            key = module.f_key(l) * l_mask
            value = module.f_value(l) * l_mask
            n_l = value.size(-1)
            query = query.reshape(
                B, HW, module.num_heads, module.key_channels // module.num_heads
            ).permute(0, 2, 1, 3)
            key = key.reshape(
                B, module.num_heads, module.key_channels // module.num_heads, n_l
            )
            value = value.reshape(
                B, module.num_heads, module.value_channels // module.num_heads, n_l
            )
            l_mask_4d = l_mask.unsqueeze(1)
            sim_map = torch.matmul(query, key)
            sim_map = (module.key_channels ** -0.5) * sim_map
            sim_map = sim_map + (1e4 * l_mask_4d - 1e4)
            sim_map = F.softmax(sim_map, dim=-1)
            extractor.pos_attn_maps[stage_idx] = sim_map.detach().cpu()

            nega_key = module.f_key(nl) * nl_mask
            nega_value = module.f_value(nl) * nl_mask
            nega_key = nega_key.reshape(
                B, module.num_heads, module.key_channels // module.num_heads, n_l
            )
            nega_value = nega_value.reshape(
                B, module.num_heads, module.value_channels // module.num_heads, n_l
            )
            nl_mask_4d = nl_mask.unsqueeze(1)
            nega_sim_map = torch.matmul(query, nega_key)
            nega_sim_map = (module.key_channels ** -0.5) * nega_sim_map
            nega_sim_map = nega_sim_map + (1e4 * nl_mask_4d - 1e4)
            nega_sim_map = F.softmax(nega_sim_map, dim=-1)
            extractor.neg_attn_maps[stage_idx] = nega_sim_map.detach().cpu()

            positive_att = torch.matmul(sim_map, value.permute(0, 1, 3, 2))
            negative_att = torch.matmul(nega_sim_map, nega_value.permute(0, 1, 3, 2))

            base = positive_att
            diff = positive_att - negative_att
            base = (
                base.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B, HW, module.value_channels)
                .permute(0, 2, 1)
            )
            diff = (
                diff.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B, HW, module.value_channels)
                .permute(0, 2, 1)
            )
            gate = module.causalGate(base)
            out = base + gate * diff
            extractor.base_gate_feats[stage_idx] = out.detach().cpu()
            out = module.W(out)
            out = out.permute(0, 2, 1)
            return out

        module.forward = forward

    @staticmethod
    def _to_spatial(attn_4d, token_mask):
        attn = attn_4d * token_mask.unsqueeze(1).unsqueeze(2)
        spatial = attn.max(dim=-1).values.mean(dim=1).squeeze(0)
        h, w = _infer_hw_from_count(spatial.numel())
        return spatial.view(h, w)

    def spatial_attn(self, stage, l_mask_cpu, nl_mask_cpu=None, source='base_gate'):
        if source == 'base_gate':
            feat = self.base_gate_feats[stage]  # (B, C, HW)
            spatial = feat.norm(dim=1).squeeze(0)
            h, w = _infer_hw_from_count(spatial.numel())
            return spatial.view(h, w)

        pos = self._to_spatial(self.pos_attn_maps[stage], l_mask_cpu.squeeze(-1).float())
        if source == 'positive':
            return pos
        if nl_mask_cpu is None:
            return pos
        neg = self._to_spatial(self.neg_attn_maps[stage], nl_mask_cpu.squeeze(-1).float())
        if source == 'negative':
            return neg
        return (pos - neg).clamp(min=0.0)

    def combined_spatial_attn(self, l_mask_cpu, nl_mask_cpu, target_hw, source='base_gate'):
        combined = None
        for s in sorted(self.pos_attn_maps.keys()):
            a = self.spatial_attn(s, l_mask_cpu, nl_mask_cpu, source=source)
            a = F.interpolate(
                a[None, None], size=target_hw, mode='bilinear', align_corners=False
            ).squeeze()
            combined = a if combined is None else combined + a
        return combined / max(len(self.pos_attn_maps), 1)

    def clear(self):
        self.pos_attn_maps.clear()
        self.neg_attn_maps.clear()
        self.base_gate_feats.clear()


def get_dataset(image_set, transform, args):
    if getattr(args, 'rrsis_dataset', 'refsegrs') == 'rrsisd':
        from data.bidataset_refer_bert import ReferDataset
    else:
        from data.binewdataset_refer_bert import ReferDataset

    ds = ReferDataset(
        args,
        split=image_set,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=True,
    )
    return ds, 2


def get_transform(args):
    return T.Compose(
        [
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def deletion_insertion_auc_cafi(
    model,
    bert_model,
    image,
    sentences_j,
    attentions_j,
    negative_sentences_j,
    negative_attentions_j,
    attn_map,
    gt_mask,
    device,
    n_steps=20,
):
    gt = gt_mask.squeeze().numpy()
    a = F.interpolate(attn_map[None, None], size=gt.shape, mode='bilinear', align_corners=False).squeeze().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    order = np.argsort(a.flatten())[::-1]
    N = len(order)
    step = max(N // n_steps, 1)
    img_np = image.cpu().squeeze(0).numpy()

    def _run(img_tensor):
        with torch.no_grad():
            if bert_model is not None:
                lhs = bert_model(sentences_j, attention_mask=attentions_j)[0]
                emb = lhs.permute(0, 2, 1)
                n_lhs = bert_model(negative_sentences_j, attention_mask=negative_attentions_j)[0]
                n_emb = n_lhs.permute(0, 2, 1)
                out = model(
                    img_tensor,
                    emb,
                    l_mask=attentions_j.unsqueeze(-1),
                    nl_feats=n_emb,
                    nl_mask=negative_attentions_j.unsqueeze(-1),
                )
            else:
                out = model(
                    img_tensor,
                    sentences_j,
                    l_mask=attentions_j,
                    nl_feats=negative_sentences_j,
                    nl_mask=negative_attentions_j,
                )
        pred = out.cpu().argmax(1).data.numpy()
        I = np.sum(np.logical_and(pred, gt))
        U = np.sum(np.logical_or(pred, gt))
        return I / (U + 1e-8)

    base_iou = _run(image)
    del_scores, ins_scores = [base_iou], [0.0]
    for s in range(1, n_steps + 1):
        n_px = min(s * step, N)
        idxs = order[:n_px]
        rows, cols = idxs // gt.shape[1], idxs % gt.shape[1]

        del_img = img_np.copy()
        del_img[:, rows, cols] = 0
        del_scores.append(_run(torch.tensor(del_img, dtype=torch.float32)[None].to(device)))

        ins_img = np.zeros_like(img_np)
        ins_img[:, rows, cols] = img_np[:, rows, cols]
        ins_scores.append(_run(torch.tensor(ins_img, dtype=torch.float32)[None].to(device)))

    x = np.linspace(0, 1, n_steps + 1)
    return float(np.trapz(del_scores, x)), float(np.trapz(ins_scores, x))


def evaluate_interpretability_cafi(model, data_loader, bert_model, device, args):
    model.eval()
    use_pwam = args.interp_saliency == 'pwam'
    extractor = CAFIAttentionExtractor(model) if use_pwam else None
    dec_hook = (
        __import__('eval_interpretability').DecoderL2Hook(model.classifier)
        if args.interp_saliency == 'decoder_l2'
        else None
    )
    metric_logger = utils.MetricLogger(delimiter="  ")

    combined_keys = ['pga', 'aiou', 'eir', 'sae', 'msc', 'trs', 'iou']
    stage_metric_keys = ['pga', 'aiou', 'eir', 'sae']
    results = {k: [] for k in combined_keys}
    stage_results = {s: {k: [] for k in stage_metric_keys} for s in range(4)}
    if args.faithfulness:
        results['del_auc'] = []
        results['ins_auc'] = []

    random_pga, random_eir = [], []
    target_hw = (args.img_size // 4, args.img_size // 4)
    n_samples = 0

    with torch.no_grad():
        for _, data in enumerate(metric_logger.log_every(data_loader, 100, 'InterpretabilityEvalCAFI:')):
            image, target, sentences, attentions, negative_sentences, negative_attentions = data
            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device).squeeze(1)
            attentions = attentions.to(device).squeeze(1)
            negative_sentences = negative_sentences.to(device).squeeze(1)
            negative_attentions = negative_attentions.to(device).squeeze(1)
            gt_np = target.cpu().data.numpy()
            gt_tensor = torch.tensor(gt_np).float()

            if gt_np.sum() == 0:
                continue

            for j in range(sentences.size(-1)):
                l_mask_cpu = attentions[:, :, j].unsqueeze(-1).cpu()
                nl_mask_cpu = negative_attentions[:, :, j].unsqueeze(-1).cpu()

                if bert_model is not None:
                    lhs = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    emb = lhs.permute(0, 2, 1)
                    n_lhs = bert_model(
                        negative_sentences[:, :, j],
                        attention_mask=negative_attentions[:, :, j],
                    )[0]
                    n_emb = n_lhs.permute(0, 2, 1)
                    output = model(
                        image,
                        emb,
                        l_mask=attentions[:, :, j].unsqueeze(-1),
                        nl_feats=n_emb,
                        nl_mask=negative_attentions[:, :, j].unsqueeze(-1),
                    )
                else:
                    output = model(
                        image,
                        sentences[:, :, j],
                        l_mask=attentions[:, :, j],
                        nl_feats=negative_sentences[:, :, j],
                        nl_mask=negative_attentions[:, :, j],
                    )

                output_mask = output.cpu().argmax(1).data.numpy()
                I, U = compute_iou(output_mask, gt_np)
                results['iou'].append(I / (U + 1e-8))

                if args.interp_saliency == 'logits_fg':
                    combined = saliency_logits_fg(output)
                elif args.interp_saliency == 'decoder_l2':
                    combined = saliency_decoder_l2(dec_hook)
                else:
                    combined = extractor.combined_spatial_attn(
                        l_mask_cpu,
                        nl_mask_cpu,
                        target_hw,
                        source=args.cafi_attn_source,
                    )

                if use_pwam:
                    for s in range(4):
                        if s not in extractor.pos_attn_maps:
                            continue
                        a_s = extractor.spatial_attn(
                            s,
                            l_mask_cpu,
                            nl_mask_cpu,
                            source=args.cafi_attn_source,
                        )
                        stage_results[s]['pga'].append(pointing_game(a_s, gt_tensor))
                        stage_results[s]['aiou'].append(attention_iou(a_s, gt_tensor))
                        stage_results[s]['eir'].append(energy_inside_ratio(a_s, gt_tensor))
                        stage_results[s]['sae'].append(spatial_entropy(a_s))

                results['pga'].append(pointing_game(combined, gt_tensor))
                results['aiou'].append(attention_iou(combined, gt_tensor))
                results['eir'].append(energy_inside_ratio(combined, gt_tensor))
                results['sae'].append(spatial_entropy(combined))

                if use_pwam:
                    msc_val, _ = multiscale_consistency(
                        extractor,
                        l_mask_cpu,
                        target_hw,
                    )
                    results['msc'].append(msc_val)
                    if args.cafi_attn_source == 'positive':
                        trs = token_relevance_score(extractor, 3, gt_tensor, l_mask_cpu)
                        if trs is not None:
                            results['trs'].append(trs)

                gt_ratio = float(gt_tensor.sum() / gt_tensor.numel())
                random_pga.append(gt_ratio)
                random_eir.append(gt_ratio)

                if args.faithfulness and n_samples < args.faithfulness_samples:
                    d_auc, i_auc = deletion_insertion_auc_cafi(
                        model,
                        bert_model,
                        image,
                        sentences[:, :, j],
                        attentions[:, :, j],
                        negative_sentences[:, :, j],
                        negative_attentions[:, :, j],
                        combined,
                        gt_tensor,
                        device,
                        n_steps=args.faithfulness_steps,
                    )
                    results['del_auc'].append(d_auc)
                    results['ins_auc'].append(i_auc)

                n_samples += 1
                if extractor is not None:
                    extractor.clear()

    if dec_hook is not None:
        dec_hook.remove()

    report = _aggregate(results, stage_results, random_pga, random_eir, n_samples, args)
    report['cafi_attn_source'] = args.cafi_attn_source
    return report


def build_parser():
    from args import get_parser

    parser = get_parser()
    parser.add_argument(
        '--cafi_attn_source',
        default='base_gate',
        choices=['base_gate', 'positive', 'negative', 'diff'],
        help='PWAM source for CAFI attention map',
    )
    return parser


def main(args):
    if args.compare:
        if len(args.compare_files) != 2:
            print('Error: --compare requires exactly 2 JSON paths.')
            sys.exit(1)
        compare_reports(args.compare_files[0], args.compare_files[1])
        return

    args.model = 'lavt_catt'
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    print(f'Model: {args.model}')
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    ckpt = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(ckpt['model'], strict=False)
    model = single_model.to(device)

    bert_model = None
    single_bert = BertModel.from_pretrained(args.ck_bert)
    if args.ddp_trained_weights:
        single_bert.pooler = None
        bert_state = {k.replace('module.', ''): v for k, v in ckpt['bert_model'].items()}
    else:
        bert_state = ckpt['bert_model']
    single_bert.load_state_dict(bert_state, strict=False)
    bert_model = single_bert.to(device)

    report = evaluate_interpretability_cafi(model, data_loader, bert_model, device, args)
    print_report(report)

    os.makedirs(args.output_json_dir, exist_ok=True)
    out_name = (
        args.output_json_name
        or f'interp_{args.model}_{args.dataset}_{args.split}_{args.cafi_attn_source}.json'
    )
    out_path = os.path.join(args.output_json_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
