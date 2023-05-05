import os

import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt


def compute_avg_bbox(args):
    mask_dir = os.path.join(args.base_dir, 'training', 'gt_image')

    all_bbox = np.empty((0, 4))
    im_sizes = np.empty((0, 2))
    for i,mask_file in enumerate(os.listdir(mask_dir)):
        # if '.png' not in mask_file: continue

        gt_mask = imageio.imread(os.path.join(mask_dir, mask_file))
        gt_mask = torch.tensor(gt_mask)[...,2] / 255.

        im_sizes = np.vstack((im_sizes, np.array(gt_mask.shape[::-1])))

        x,y = torch.where(gt_mask == 1)
        all_bbox = np.vstack((all_bbox, np.array([[y.min(), x.min(), y.max(), x.max()]])))

    save = np.hstack(([np.mean(im_sizes, axis=0).astype(int), np.mean(all_bbox, axis=0)]))
    np.save(os.path.join(args.output_dir, 'avg_bboxes.npy'), save)


def draw_mask_onimage(X, mask, path):
    mask = mask.detach().cpu().numpy()
    plt.figure()
    plt.imshow(X)
    color = np.array([255/255, 50/255, 50/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask)
    plt.savefig(path)
