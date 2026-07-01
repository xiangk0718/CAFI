#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPUS=${GPUS:-0,1}
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC=${NPROC:-${#GPU_ARRAY[@]}}
MASTER_PORT=${MASTER_PORT:-12346}

mkdir -p checkpoints logs

CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --cafi_variant lgce \
  --rrsis_dataset refsegrs \
  --dataset refsegrs \
  --model lavt_catt \
  --model_id cafi_lgce_refsegrs \
  --batch-size 2 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert ./pretrained_weights/bert \
  --bert_tokenizer ./pretrained_weights/bert \
  --epochs 60 \
  --img_size 480 \
  --window12 \
  --workers 0 \
  --swap_aug_ratio 0.02 \
  "$@" \
  2>&1 | tee logs/train_cafi_lgce_refsegrs.log
