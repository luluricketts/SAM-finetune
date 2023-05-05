import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np

from src import SAMWrapper, train, test, compute_avg_bbox

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, required=True, help='dataset name under data/')
parser.add_argument('--mode', type=str, required=True, help='train | test | bbox')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--ckpt_every', type=int, default=10)
parser.add_argument('--save_every', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ckpt_name', type=str, default=None)
if __name__ == "__main__":
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {args.device}')

    args.base_dir = os.path.join('data', args.base_dir)
    args.output_dir = os.path.join('logs', args.exp_name)
    args.checkpoint_dir = os.path.join('checkpoint', args.exp_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.mode == 'train':
        print('TRAIN MODE')
        args.results_dir = os.path.join(args.output_dir, 'train_results')
        os.makedirs(args.results_dir, exist_ok=True)
        
        if args.ckpt_name:
            model = SAMWrapper(os.path.join(args.checkpoint_dir, args.ckpt_name), args.device)
        else:
            print('Finetuning from SAM checkpoint and reinitializing MLP parameters')
            model = SAMWrapper(os.path.join('checkpoint', 'sam_vit_h_4b8939.pth'), args.device, from_scratch=True)
        model = model.to(args.device).train()
        train(args, model)

    elif args.mode == 'test':
        print('TEST MODE')
        args.results_dir = os.path.join(args.output_dir, 'test_results')
        os.makedirs(args.results_dir, exist_ok=True)

        try:
            avg_bbox = np.load(os.path.join(args.output_dir, 'avg_bboxes.npy'))
        except:
            avg_bbox = None
            print('No average bboxes available. Run with `--mode bbox` first for better results')
        if args.ckpt_name:
            model = SAMWrapper(os.path.join(args.checkpoint_dir, args.ckpt_name), args.device, avg_box=avg_bbox)
        else:
            print('Loading best model')
            model = SAMWrapper(os.path.join(args.checkpoint_dir, 'best_model.pth'), args.device, avg_box=avg_bbox)
        model = model.to(args.device).eval()
        test(args, model)
 
    elif args.mode == 'bbox':
        os.makedirs(args.output_dir, exist_ok=True)
        compute_avg_bbox(args)
        
    else:
        print(f'{args.mode} not supported, please specify mode [train | test | bbox]')
