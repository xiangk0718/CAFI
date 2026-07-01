"""
Quantitative Interpretability Evaluation for LAVT-RIS.

Extracts cross-modal attention maps from PWAM (SpatialImageLanguageAttention)
modules and computes reproducible metrics against ground-truth masks.

Metrics
-------
  PGA   : Pointing Game Accuracy
  AIoU  : Attention-Ground Truth IoU (best over thresholds)
  EIR   : Energy Inside Ratio
  SAE   : Spatial Attention Entropy (normalized, lower = more focused)
  MSC   : Multi-Scale Consistency (Pearson r between stage attention maps)
  TRS   : Token Relevance Score (GT-region / BG-region attention ratio)
  D-AUC : Deletion AUC (faithfulness, optional)
  I-AUC : Insertion AUC (faithfulness, optional)

Usage
-----
  python eval_interpretability.py \
      --model lavt --swin_type base --dataset rrsisd --split test \
      --resume ./checkpoints/model_best_lavt_rrsisd.pth \
      --ck_bert ./pretrained_weights/bert \
      --ddp_trained_weights --window12 --img_size 480 \
      --output-json-dir ./interpretability_results \
      [--faithfulness --faithfulness_samples 100 --faithfulness_steps 20] \
      [--save_vis --vis_samples 50]

Compare two models
------------------
  python eval_interpretability.py --compare \
      result_a.json result_b.json
"""

import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from scipy import stats as sp_stats

from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils


def _infer_hw_from_count(hw_flat: int):
    """Factor HW tokens into (H, W); prefer factors near sqrt (Swin grids may be non-square)."""
    s = int(math.sqrt(hw_flat))
    while s >= 1:
        if hw_flat % s == 0:
            return s, hw_flat // s
        s -= 1
    return 1, hw_flat


# ================================================================
#  Attention Extraction
# ================================================================

class AttentionExtractor:
    """Monkey-patches SpatialImageLanguageAttention.forward to store sim_map."""

    def __init__(self, model):
        self.attn_maps = {}
        self._n_stages = len(model.backbone.layers)
        for i, layer in enumerate(model.backbone.layers):
            self._patch(layer.fusion.image_lang_att, i)

    # ---- internal ----

    def _patch(self, module, stage_idx):
        extractor = self

        def forward(x, l, l_mask):
            B, HW = x.size(0), x.size(1)
            x = x.permute(0, 2, 1)
            l_mask = l_mask.permute(0, 2, 1)

            query = module.f_query(x).permute(0, 2, 1)
            key = module.f_key(l) * l_mask
            value = module.f_value(l) * l_mask
            n_l = value.size(-1)

            query = query.reshape(
                B, HW, module.num_heads,
                module.key_channels // module.num_heads
            ).permute(0, 2, 1, 3)
            key = key.reshape(
                B, module.num_heads,
                module.key_channels // module.num_heads, n_l
            )
            value = value.reshape(
                B, module.num_heads,
                module.value_channels // module.num_heads, n_l
            )
            l_mask = l_mask.unsqueeze(1)

            sim_map = torch.matmul(query, key)
            sim_map = (module.key_channels ** -0.5) * sim_map
            sim_map = sim_map + (1e4 * l_mask - 1e4)
            sim_map = F.softmax(sim_map, dim=-1)

            extractor.attn_maps[stage_idx] = sim_map.detach().cpu()

            out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))
            out = (
                out.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B, HW, module.value_channels)
            )
            out = out.permute(0, 2, 1)
            out = module.W(out)
            out = out.permute(0, 2, 1)
            return out

        module.forward = forward

    # ---- public API ----

    def spatial_attn(self, stage, l_mask_cpu):
        """Spatial map (H, W) for *stage*, averaged over heads.

        sim_map is softmax over *language* tokens, so sum_l p(l)=1 at every (head, hw) and
        summing p(l) over l gives a flat map — useless for visualization. Use max_l p(l)
        (peak token mass at each location) so values vary in [~1/N_l, 1] across space.
        """
        attn = self.attn_maps[stage]                    # (1, heads, HW, N_l)
        mask = l_mask_cpu.squeeze(-1).float()           # (1, N_l)
        attn = attn * mask.unsqueeze(1).unsqueeze(2)
        spatial = attn.max(dim=-1).values.mean(dim=1).squeeze(0)  # (HW,)
        h, w = _infer_hw_from_count(spatial.numel())
        return spatial.view(h, w)

    def combined_spatial_attn(self, l_mask_cpu, target_hw):
        """Multi-scale fused spatial attention up-sampled to *target_hw*."""
        combined = None
        for s in sorted(self.attn_maps.keys()):
            a = self.spatial_attn(s, l_mask_cpu)
            a = F.interpolate(
                a[None, None], size=target_hw,
                mode='bilinear', align_corners=False
            ).squeeze()
            combined = a if combined is None else combined + a
        return combined / len(self.attn_maps)

    def clear(self):
        self.attn_maps.clear()


class DecoderL2Hook:
    """Captures input to classifier.conv1_1 (B,C,H,W) — last decoder features before logits."""

    def __init__(self, classifier):
        self._feat = None

        def _pre_hook(_module, inputs):
            self._feat = inputs[0].detach()

        self._handle = classifier.conv1_1.register_forward_pre_hook(_pre_hook)

    def pop(self):
        f = self._feat
        self._feat = None
        return f

    def remove(self):
        self._handle.remove()


def saliency_logits_fg(output):
    """Foreground class probability map, full input resolution (after model interpolate)."""
    prob = F.softmax(output, dim=1)[:, 1]
    return prob.squeeze(0).detach().cpu()


def saliency_decoder_l2(hook):
    """Per-pixel channel L2 norm of last decoder features."""
    feat = hook.pop()
    if feat is None:
        raise RuntimeError('DecoderL2Hook: conv1_1 did not run (hook empty).')
    t = feat.squeeze(0).flatten(1).norm(dim=0)
    H = W = int(round(np.sqrt(t.numel())))
    return t.view(H, W).cpu()


# ================================================================
#  Metric Functions
# ================================================================

def _upsample(attn_2d, target_shape):
    """Up-sample a (H, W) attention tensor to *target_shape*."""
    return F.interpolate(
        attn_2d[None, None], size=target_shape,
        mode='bilinear', align_corners=False
    ).squeeze()


def pointing_game(attn_map, gt_mask):
    """1.0 if argmax of attention falls inside GT, else 0.0."""
    gt = gt_mask.squeeze()
    a = _upsample(attn_map, gt.shape)
    idx = a.argmax()
    r, c = idx // gt.shape[1], idx % gt.shape[1]
    return 1.0 if gt[r, c] > 0 else 0.0


def attention_iou(attn_map, gt_mask):
    """Best IoU between binarized attention and GT across thresholds."""
    gt = gt_mask.squeeze().numpy()
    a = _upsample(attn_map, gt.shape).numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    best = 0.0
    for t in np.arange(0.05, 1.0, 0.05):
        pred = (a >= t).astype(np.float32)
        inter = (pred * gt).sum()
        union = np.clip(pred + gt, 0, 1).sum()
        iou = inter / (union + 1e-8)
        if iou > best:
            best = iou
    return best


def energy_inside_ratio(attn_map, gt_mask):
    """Fraction of total attention energy that falls inside GT."""
    gt = gt_mask.squeeze().float()
    a = _upsample(attn_map, gt.shape)
    return float((a * gt).sum() / (a.sum() + 1e-8))


def spatial_entropy(attn_map):
    """Normalised Shannon entropy of spatial attention (lower ⇒ more focused)."""
    flat = attn_map.view(-1)
    p = flat / (flat.sum() + 1e-8)
    p = p.clamp(min=1e-10)
    H = -(p * torch.log2(p)).sum().item()
    H_max = np.log2(flat.numel())
    return H / H_max


def multiscale_consistency(extractor, l_mask_cpu, target_hw):
    """Mean pairwise Pearson r between up-sampled stage attention maps."""
    stages = sorted(extractor.attn_maps.keys())
    if len(stages) < 2:
        return 0.0, {}
    flat = {}
    for s in stages:
        a = extractor.spatial_attn(s, l_mask_cpu)
        a = F.interpolate(
            a[None, None], size=target_hw,
            mode='bilinear', align_corners=False
        ).squeeze().numpy().flatten()
        flat[s] = a
    pairs = {}
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            r, _ = sp_stats.pearsonr(flat[stages[i]], flat[stages[j]])
            pairs[f's{stages[i]}_s{stages[j]}'] = float(r)
    return float(np.mean(list(pairs.values()))), pairs


def token_relevance_score(extractor, stage, gt_mask, l_mask_cpu):
    """Ratio: mean attention from GT-region → valid tokens vs. BG-region → valid tokens."""
    attn = extractor.attn_maps[stage]   # (1, heads, HW, N_l)
    HW = attn.shape[2]
    H = W = int(round(np.sqrt(HW)))
    gt = gt_mask.squeeze()
    gt_down = F.interpolate(
        gt.float()[None, None], size=(H, W), mode='nearest'
    ).squeeze().numpy().flatten()
    attn_avg = attn.squeeze(0).mean(0).numpy()  # (HW, N_l)
    mask = l_mask_cpu.squeeze().numpy()
    if mask.ndim > 1:
        mask = mask.squeeze(-1)
    gt_px = gt_down > 0
    bg_px = gt_down == 0
    if gt_px.sum() == 0 or bg_px.sum() == 0:
        return None
    valid = mask > 0
    gt_attn = attn_avg[gt_px][:, valid].mean()
    bg_attn = attn_avg[bg_px][:, valid].mean()
    return float(gt_attn / (bg_attn + 1e-8))


def deletion_insertion_auc(
    model, bert_model, image, sentences_j, attentions_j,
    attn_map, gt_mask, device, n_steps=20
):
    """Faithfulness: Deletion-AUC and Insertion-AUC."""
    gt = gt_mask.squeeze().numpy()
    a = _upsample(attn_map, gt.shape).numpy()
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
                out = model(img_tensor, emb, l_mask=attentions_j.unsqueeze(-1))
            else:
                out = model(img_tensor, sentences_j, l_mask=attentions_j)
        pred = out.cpu().argmax(1).data.numpy()
        I = np.sum(np.logical_and(pred, gt))
        U = np.sum(np.logical_or(pred, gt))
        return I / (U + 1e-8)

    base_iou = _run(image)
    del_scores = [base_iou]
    ins_scores = [0.0]

    for s in range(1, n_steps + 1):
        n_px = min(s * step, N)
        idxs = order[:n_px]
        rows, cols = idxs // gt.shape[1], idxs % gt.shape[1]

        del_img = img_np.copy()
        del_img[:, rows, cols] = 0
        del_scores.append(
            _run(torch.tensor(del_img, dtype=torch.float32)[None].to(device))
        )

        ins_img = np.zeros_like(img_np)
        ins_img[:, rows, cols] = img_np[:, rows, cols]
        ins_scores.append(
            _run(torch.tensor(ins_img, dtype=torch.float32)[None].to(device))
        )

    x = np.linspace(0, 1, n_steps + 1)
    return float(np.trapz(del_scores, x)), float(np.trapz(ins_scores, x))


# ================================================================
#  Visualization helpers (optional)
# ================================================================

def save_attn_overlay(image_tensor, attn_map, gt_mask, save_path):
    """Save attention heat-map overlaid on the original image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    gt = gt_mask.squeeze().numpy()
    a = _upsample(attn_map, gt.shape).numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a_resized = cv2.resize(a, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * a_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay)


# ================================================================
#  Dataset / Transform / IoU  (reused from existing code)
# ================================================================

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(
        args, split=image_set,
        image_transforms=transform, target_transforms=None,
        eval_mode=True,
    )
    return ds, 2


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_iou(pred, gt):
    I = np.sum(np.logical_and(pred, gt))
    U = np.sum(np.logical_or(pred, gt))
    return I, U


# ================================================================
#  Main Evaluation Loop
# ================================================================

def evaluate_interpretability(model, data_loader, bert_model, device, args):
    model.eval()
    use_pwam = args.interp_saliency == 'pwam'
    extractor = AttentionExtractor(model) if use_pwam else None
    dec_hook = (
        DecoderL2Hook(model.classifier)
        if args.interp_saliency == 'decoder_l2'
        else None
    )
    metric_logger = utils.MetricLogger(delimiter="  ")

    combined_keys = ['pga', 'aiou', 'eir', 'sae', 'msc', 'trs', 'iou']
    stage_metric_keys = ['pga', 'aiou', 'eir', 'sae']
    results = {k: [] for k in combined_keys}
    stage_results = {
        s: {k: [] for k in stage_metric_keys}
        for s in range(4)
    }
    if args.faithfulness:
        results['del_auc'] = []
        results['ins_auc'] = []

    random_pga, random_eir = [], []
    target_hw = (args.img_size // 4, args.img_size // 4)
    n_samples = 0
    vis_dir = None
    if args.save_vis:
        vis_dir = os.path.join(args.output_json_dir, 'vis_attn')
        os.makedirs(vis_dir, exist_ok=True)

    header = 'InterpretabilityEval:'
    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, sentences, attentions = data
            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device).squeeze(1)
            attentions = attentions.to(device).squeeze(1)
            gt_np = target.cpu().data.numpy()
            gt_tensor = torch.tensor(gt_np).float()

            if gt_np.sum() == 0:
                continue

            for j in range(sentences.size(-1)):
                l_mask_cpu = attentions[:, :, j].unsqueeze(-1).cpu()

                if bert_model is not None:
                    lhs = bert_model(
                        sentences[:, :, j],
                        attention_mask=attentions[:, :, j],
                    )[0]
                    emb = lhs.permute(0, 2, 1)
                    output = model(
                        image, emb,
                        l_mask=attentions[:, :, j].unsqueeze(-1),
                    )
                else:
                    output = model(
                        image, sentences[:, :, j],
                        l_mask=attentions[:, :, j],
                    )

                output_mask = output.cpu().argmax(1).data.numpy()
                I, U = compute_iou(output_mask, gt_np)
                results['iou'].append(I / (U + 1e-8))

                if args.interp_saliency == 'logits_fg':
                    combined = saliency_logits_fg(output)
                elif args.interp_saliency == 'decoder_l2':
                    combined = saliency_decoder_l2(dec_hook)
                else:
                    combined = extractor.combined_spatial_attn(l_mask_cpu, target_hw)

                # --- per-stage metrics (PWAM only) ---
                if use_pwam:
                    for s in range(4):
                        if s not in extractor.attn_maps:
                            continue
                        a_s = extractor.spatial_attn(s, l_mask_cpu)
                        stage_results[s]['pga'].append(pointing_game(a_s, gt_tensor))
                        stage_results[s]['aiou'].append(attention_iou(a_s, gt_tensor))
                        stage_results[s]['eir'].append(energy_inside_ratio(a_s, gt_tensor))
                        stage_results[s]['sae'].append(spatial_entropy(a_s))

                # --- combined metrics ---
                results['pga'].append(pointing_game(combined, gt_tensor))
                results['aiou'].append(attention_iou(combined, gt_tensor))
                results['eir'].append(energy_inside_ratio(combined, gt_tensor))
                results['sae'].append(spatial_entropy(combined))

                if use_pwam:
                    msc_val, _ = multiscale_consistency(extractor, l_mask_cpu, target_hw)
                    results['msc'].append(msc_val)

                    trs = token_relevance_score(extractor, 3, gt_tensor, l_mask_cpu)
                    if trs is not None:
                        results['trs'].append(trs)

                # --- random baseline ---
                gt_ratio = float(gt_tensor.sum() / gt_tensor.numel())
                random_pga.append(gt_ratio)
                random_eir.append(gt_ratio)

                # --- faithfulness (optional, expensive) ---
                if args.faithfulness and n_samples < args.faithfulness_samples:
                    d_auc, i_auc = deletion_insertion_auc(
                        model, bert_model, image,
                        sentences[:, :, j], attentions[:, :, j],
                        combined, gt_tensor, device,
                        n_steps=args.faithfulness_steps,
                    )
                    results['del_auc'].append(d_auc)
                    results['ins_auc'].append(i_auc)

                # --- save visualisation ---
                if vis_dir and n_samples < args.vis_samples:
                    save_attn_overlay(
                        image, combined, gt_tensor,
                        os.path.join(vis_dir, f'{idx:05d}_{j}.png'),
                    )

                n_samples += 1
                if extractor is not None:
                    extractor.clear()

            del image, target, sentences, attentions, output
            if bert_model is not None:
                del lhs, emb

    if dec_hook is not None:
        dec_hook.remove()

    report = _aggregate(results, stage_results, random_pga, random_eir,
                        n_samples, args)
    return report


# ================================================================
#  Statistics & Reporting
# ================================================================

def _stats(values):
    a = np.array(values)
    n = len(a)
    m = float(a.mean())
    s = float(a.std())
    ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return {'mean': round(m, 4), 'std': round(s, 4),
            'ci95': round(ci, 4), 'n': n}


def _aggregate(results, stage_results, random_pga, random_eir, n_samples, args):
    report = {
        'model': args.model,
        'dataset': args.dataset,
        'split': args.split,
        'num_samples': n_samples,
        'timestamp': datetime.datetime.now().isoformat(),
        'interp_saliency': args.interp_saliency,
        'metrics': {},
        'per_stage': {},
    }

    for k in ['iou', 'pga', 'aiou', 'eir', 'sae', 'msc', 'trs']:
        if results[k]:
            report['metrics'][k] = _stats(results[k])

    for s in range(4):
        stage_d = {}
        for k in ['pga', 'aiou', 'eir', 'sae']:
            if stage_results[s][k]:
                stage_d[k] = _stats(stage_results[s][k])
        report['per_stage'][f'stage_{s}'] = stage_d

    if args.faithfulness and results.get('del_auc'):
        report['metrics']['del_auc'] = _stats(results['del_auc'])
        report['metrics']['ins_auc'] = _stats(results['ins_auc'])

    if random_pga:
        rand_pga = float(np.mean(random_pga))
        rand_eir = float(np.mean(random_eir))
        pga_arr = np.array(results['pga'])
        eir_arr = np.array(results['eir'])
        t_pga, p_pga = sp_stats.ttest_1samp(pga_arr, rand_pga)
        t_eir, p_eir = sp_stats.ttest_1samp(eir_arr, rand_eir)
        report['baseline_comparison'] = {
            'random_pga': round(rand_pga, 4),
            'random_eir': round(rand_eir, 4),
            'pga_vs_random_t': round(float(t_pga), 4),
            'pga_vs_random_p': round(float(p_pga), 6),
            'eir_vs_random_t': round(float(t_eir), 4),
            'eir_vs_random_p': round(float(p_eir), 6),
        }

    return report


_METRIC_NAMES = {
    'iou':     'Segmentation IoU',
    'pga':     'Pointing Game Acc.',
    'aiou':    'Attention IoU',
    'eir':     'Energy Inside Ratio',
    'sae':     'Spatial Attn Entropy',
    'msc':     'Multi-scale Consist.',
    'trs':     'Token Relevance Score',
    'del_auc': 'Deletion AUC',
    'ins_auc': 'Insertion AUC',
}


def print_report(report):
    w = 70
    print('\n' + '=' * w)
    print('  QUANTITATIVE INTERPRETABILITY EVALUATION REPORT')
    print('=' * w)
    print(f"  Model : {report['model']}  |  Dataset: {report['dataset']}"
          f"  |  Split: {report['split']}")
    print(f"  Samples: {report['num_samples']}  |  {report['timestamp']}")
    sal = report.get('interp_saliency', 'pwam')
    print(f"  Saliency source: {sal}")
    print('-' * w)

    print('\n  [Combined Metrics]')
    for k, name in _METRIC_NAMES.items():
        if k in report['metrics']:
            m = report['metrics'][k]
            print(f"    {name:<25s}: {m['mean']:.4f} "
                  f"\u00b1 {m['std']:.4f}  (95%CI: \u00b1{m['ci95']:.4f}, n={m['n']})")

    print('\n  [Per-Stage Metrics]')
    if report.get('interp_saliency', 'pwam') != 'pwam':
        print('    (omitted — only defined for --interp_saliency pwam)')
    else:
        hdr = f"    {'Stage':<8s}"
        for k in ['pga', 'aiou', 'eir', 'sae']:
            hdr += f" {_METRIC_NAMES[k]:>20s}"
        print(hdr)
        print('    ' + '-' * 88)
        for s in range(4):
            key = f'stage_{s}'
            if key not in report['per_stage']:
                continue
            row = f'    S{s:<7d}'
            for k in ['pga', 'aiou', 'eir', 'sae']:
                v = report['per_stage'][key].get(k, {}).get('mean', 0)
                row += f' {v:>20.4f}'
            print(row)

    if 'baseline_comparison' in report:
        bc = report['baseline_comparison']
        print('\n  [Statistical Test vs Random Baseline]')
        pga_m = report['metrics']['pga']['mean']
        eir_m = report['metrics']['eir']['mean']
        print(f"    PGA : model={pga_m:.4f}  random={bc['random_pga']:.4f}"
              f"  t={bc['pga_vs_random_t']:.3f}  p={bc['pga_vs_random_p']:.2e}")
        print(f"    EIR : model={eir_m:.4f}  random={bc['random_eir']:.4f}"
              f"  t={bc['eir_vs_random_t']:.3f}  p={bc['eir_vs_random_p']:.2e}")

    print('\n' + '=' * w)


# ================================================================
#  Compare Two Reports
# ================================================================

def compare_reports(path_a, path_b):
    """Load two JSON reports and print a side-by-side comparison table."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    w = 78
    print('\n' + '=' * w)
    print('  INTERPRETABILITY COMPARISON')
    print('=' * w)
    print(f"  A: {a['model']} on {a['dataset']}  ({a['num_samples']} samples)")
    print(f"  B: {b['model']} on {b['dataset']}  ({b['num_samples']} samples)")
    print('-' * w)

    print(f"\n    {'Metric':<25s} {'Model A':>10s} {'Model B':>10s} {'Delta':>10s}")
    print('    ' + '-' * 55)
    for k, name in _METRIC_NAMES.items():
        va = a['metrics'].get(k, {}).get('mean')
        vb = b['metrics'].get(k, {}).get('mean')
        if va is None or vb is None:
            continue
        delta = vb - va
        sign = '+' if delta >= 0 else ''
        print(f"    {name:<25s} {va:>10.4f} {vb:>10.4f} {sign}{delta:>9.4f}")

    if 'baseline_comparison' in a and 'baseline_comparison' in b:
        print(f"\n    {'Stat-test vs random':<25s} {'p-val A':>10s} {'p-val B':>10s}")
        print('    ' + '-' * 45)
        pa = a['baseline_comparison']['pga_vs_random_p']
        pb = b['baseline_comparison']['pga_vs_random_p']
        print(f"    {'PGA t-test p-value':<25s} {pa:>10.2e} {pb:>10.2e}")

    print('\n' + '=' * w)


# ================================================================
#  Entry Point
# ================================================================

def main(args):
    if args.compare:
        if len(args.compare_files) != 2:
            print('Error: --compare requires exactly 2 JSON paths.')
            sys.exit(1)
        compare_reports(args.compare_files[0], args.compare_files[1])
        return

    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
    )

    print(f'Model: {args.model}')
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    ckpt = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(ckpt['model'], strict=False)
    model = single_model.to(device)

    bert_model = None
    if args.model != 'lavt_one':
        single_bert = BertModel.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert.pooler = None
            bert_state = {
                k.replace('module.', ''): v
                for k, v in ckpt['bert_model'].items()
            }
        else:
            bert_state = ckpt['bert_model']
        single_bert.load_state_dict(bert_state)
        bert_model = single_bert.to(device)

    report = evaluate_interpretability(model, data_loader, bert_model, device, args)
    print_report(report)

    os.makedirs(args.output_json_dir, exist_ok=True)
    out_name = args.output_json_name or f'interp_{args.model}_{args.dataset}_{args.split}.json'
    out_path = os.path.join(args.output_json_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    from args import get_parser

    parser = get_parser()

    args = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
