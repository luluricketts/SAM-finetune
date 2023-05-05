import os

from tqdm.auto import tqdm
import torch
import torch.nn as nn

from .data import get_dataloader
from .utils import draw_mask_onimage


def train(args, model):

    dataloader = get_dataloader(os.path.join(args.base_dir, 'training'), args.mode)
    optimizer = torch.optim.Adam(model.sam_model.mask_decoder.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    accum_iter = 10
    best_model_loss = 1e10
    best_model_ckpt = None
    best_model_epoch = 0
    for ep in range(1, args.epochs + 1):
        
        total_loss = 0
        for i,(X,gt_mask) in enumerate(tqdm(dataloader)):

            X_orig = X.copy()
            gt_mask, pred_mask = model(X, gt_mask)

            # train step
            loss = loss_fn(pred_mask.squeeze(), gt_mask)
            total_loss += loss.item()
            loss.backward()

            if (i + 1) % accum_iter == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            if i % args.save_every == 0:
                draw_mask_onimage(X_orig, pred_mask.squeeze(), os.path.join(args.results_dir, f'ep{ep}_{i}.jpg'))
                draw_mask_onimage(X_orig, gt_mask, os.path.join(args.results_dir, f'ep{ep}_{i}_gt.jpg'))
            
            print(f'LOSS {loss.item()}')

            del gt_mask, pred_mask, loss
            torch.cuda.empty_cache()

        if ep % args.ckpt_every == 0:
            torch.save(model.sam_model.state_dict(), os.path.join(args.checkpoint_dir, f'sam_ckpt_{ep}.pth'))

        avg_loss = total_loss / len(dataloader.dataset)
        print(f'EPOCH {ep} | AVERAGE LOSS {avg_loss}')
        if avg_loss < best_model_loss:
            best_model_loss = avg_loss
            best_model_ckpt = model.sam_model.state_dict().copy()
            best_model_epoch = ep

    torch.save(best_model_ckpt, os.path.join(args.checkpoint_dir, f'best_model.pth'))
    print(f'BEST MODEL EPOCH {best_model_epoch} | LOSS {best_model_loss}')
