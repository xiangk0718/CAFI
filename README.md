# CAFI

Code for **"CAFI: Counterfactual Alignment and Front-Door Intervention Training for Deconfounded Interpretable Referring Remote Sensing Image Segmentation"**.

Unified codebase for training and evaluating two CAFI variants:

## Layout

```text
CAFI/
  args.py                    # shared CLI arguments
  train.py                   # unified training entry
  train_cafi_lgce_refsegrs.sh
  train_cafi_rmsin_rrsisd.sh
  test.py                    # unified test entry
  eval_interpretability.py   # CAFI interpretability metrics
  bert/ data/ loss/          # shared code
  variants/
    lgce/                    # CAFI-LGCE model code
    rmsin/                   # CAFI-RMSIN model code
  checkpoints/               # empty placeholder for trained checkpoints
  pretrained_weights/        # empty placeholder for BERT/Swin weights
  refer/                     # empty placeholder for REFER-style data
```

## Model Selection

Use `--cafi_variant`:

```bash
--cafi_variant lgce   # or CAFI-LGCE
--cafi_variant rmsin  # or CAFI-RMSIN
```

Both variants use the `lavt_catt` entry name; `train.py` selects the correct
`variants/<name>/lib` implementation before importing `lib.segmentation`.

## Dataset Selection

Use `--rrsis_dataset`:

```bash
--rrsis_dataset refsegrs
--rrsis_dataset rrsisd
```

For `refsegrs`, `data.binewdataset_refer_bert` is used. For `rrsisd`,
`data.bidataset_refer_bert` is used.

## Swapped AUG Filtering

`train.py` includes swapped positive/negative augmentation with filtering:

- RRSIS-D: only large fixed target categories from `train_catt_augv2.py`
  are eligible.
- RefSegRS: expressions containing `van`, `bus`, or `road marking` are
  excluded from the swapped augmentation set.
- All datasets: negative caption length must be no longer than
  `--swap_aug_max_nega_tokens` (default `20`).

Control the amount with:

```bash
--swap_aug_ratio 0.02
```

## Example Training

Recommended scripts:

```bash
# CAFI-LGCE on RefSegRS
GPUS=0 ./train_cafi_lgce_refsegrs.sh

# CAFI-RMSIN on RRSIS-D
GPUS=0 ./train_cafi_rmsin_rrsisd.sh
```

Arguments can be appended to either script, for example:

```bash
GPUS=0,1 NPROC=2 ./train_cafi_lgce_refsegrs.sh --model_id my_run --epochs 80
```

Equivalent raw command:

```bash
python -m torch.distributed.launch --nproc_per_node 1 train.py \
  --cafi_variant lgce \
  --rrsis_dataset refsegrs \
  --model_id cafi_lgce_refsegrs \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert ./pretrained_weights/bert \
  --bert_tokenizer ./pretrained_weights/bert \
  --epochs 40 \
  --img_size 480 \
  --window12
```

Switch to RMSIN:

```bash
python -m torch.distributed.launch --nproc_per_node 1 train.py \
  --cafi_variant rmsin \
  --rrsis_dataset refsegrs \
  --model_id cafi_rmsin_refsegrs \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert ./pretrained_weights/bert \
  --bert_tokenizer ./pretrained_weights/bert \
  --epochs 40 \
  --img_size 480 \
  --window12
```

## Test

```bash
python test.py \
  --cafi_variant lgce \
  --rrsis_dataset refsegrs \
  --resume ./checkpoints/model_best_xxx.pth \
  --ck_bert ./pretrained_weights/bert \
  --bert_tokenizer ./pretrained_weights/bert \
  --window12 \
  --img_size 480
```

## Notes

Place external assets locally before running:

- BERT weights under `pretrained_weights/bert/`
- Swin weights under `pretrained_weights/`
- trained checkpoints under `checkpoints/`
- REFER-style data under `refer/` or configure paths in dataset files.

## Pretrained Weights

Google Drive: **TODO: paste checkpoint / pretrained-weight link here**.

Expected layout after download:

```text
CAFI/
  pretrained_weights/
    bert/
    swin_base_patch4_window12_384_22k.pth
  checkpoints/
    model_best_*.pth
```

## Acknowledgements

This repository builds on and adapts code from the following projects:

- [LAVT](https://github.com/yz93/LAVT-RIS)
- RRSIS / RefSegRS
- RMSIN

We thank the authors of these repositories and datasets for making their work
available to the community.
