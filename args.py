import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='CAFI training and testing')
    parser.add_argument('--cafi_variant', default='lgce',
                        choices=['lgce', 'rmsin', 'CAFI-LGCE', 'CAFI-RMSIN'],
                        help='model code path to use: CAFI-LGCE or CAFI-RMSIN')
    parser.add_argument('--rrsis_dataset', default='refsegrs',
                        choices=['refsegrs', 'rrsisd'],
                        help='dataset loader family used by CAFI')
    parser.add_argument('--swap_aug_ratio', default=0.02, type=float,
                        help='ratio of train samples selected for swapped augmentation')
    parser.add_argument('--swap_aug_max_nega_tokens', default=20, type=int,
                        help='max valid negative-caption tokens for swapped augmentation')

    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='./pretrained_weights/bert')
    parser.add_argument('--ck_bert', default='./pretrained_weights/bert')
    parser.add_argument('--dataset', default='rrsisd',
                        help='legacy REFER dataset name kept for rrsisd loaders')
    parser.add_argument('--ddp_trained_weights', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--fusion_drop', default=0.0, type=float)
    parser.add_argument('--img_size', default=480, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--mha', default='')
    parser.add_argument('--model', default='lavt_catt')
    parser.add_argument('--model_id', default='cafi')
    parser.add_argument('--output-dir', default='./checkpoints/')
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--pretrained_swin_weights',
                        default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth')
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--refer_data_root', default='./refer/data/')
    parser.add_argument('--refsegrs_data_root', default='/irsa/irsa_xk/RefSegRS',
                        help='root path of the RefSegRS dataset')
    parser.add_argument('--resume', default='')
    parser.add_argument('--split', default='test')
    parser.add_argument('--splitBy', default='unc')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('-j', '--workers', default=0, type=int)

    # Interpretability / attention-map arguments
    parser.add_argument('--interp_saliency', default='pwam',
                        choices=['pwam', 'logits_fg', 'decoder_l2'])
    parser.add_argument('--output-json-dir', default='./interpretability_results')
    parser.add_argument('--output_json_name', default='')
    parser.add_argument('--faithfulness', action='store_true')
    parser.add_argument('--faithfulness_samples', type=int, default=100)
    parser.add_argument('--faithfulness_steps', type=int, default=20)
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--vis_samples', type=int, default=50)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('compare_files', nargs='*')
    parser.add_argument('--att_maps_dir', default='./att_maps')
    parser.add_argument('--att_maps_max', type=int, default=100)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    parser.parse_args()
