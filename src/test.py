import os

from tqdm.auto import tqdm

from .data import get_dataloader
from .utils import draw_mask_onimage

def test(args, model):
    dataloader = get_dataloader(os.path.join(args.base_dir, 'testing'), args.mode)

    for i,(X,_) in enumerate(tqdm(dataloader)):

        X_orig = X.copy()
        _, pred_mask = model(X, None)

        draw_mask_onimage(X_orig, pred_mask.squeeze(), os.path.join(args.results_dir, f'{i}.jpg'))
