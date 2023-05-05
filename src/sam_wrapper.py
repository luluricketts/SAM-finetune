import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry


class SAMWrapper(nn.Module):
    def __init__(self, ckpt_path, device, from_scratch=False, avg_box=None):
        super().__init__()
        self.device = device
        self.avg_bbox = avg_box

        self.sam_model = sam_model_registry['vit_h'](checkpoint=ckpt_path)
        if from_scratch:
            for layer in self.sam_model.mask_decoder.output_hypernetworks_mlps.children():
                for cc in layer.children():
                    for c in cc.children():
                        try:
                            c.reset_parameters()
                        except:
                            print(f'cannot reset parameters: {c}')

        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)


    def resize_bbox(self, target_size):
        x_scale = target_size[1] / self.avg_bbox[1]
        y_scale = target_size[0] / self.avg_bbox[0]

        self.avg_bbox[[2, 4]] *= y_scale
        self.avg_bbox[[3, 5]] *= x_scale
        self.avg_bbox[:2] = target_size


    def forward(self, X, gt_mask):

        # preprocessing
        original_size = X.shape[:2]
        X = self.transform.apply_image(X)
        X = torch.as_tensor(X, device=self.device)
        X = X.permute(2, 0, 1).contiguous()[None, ...]
        input_size = tuple(X.shape[-2:])
        X = self.sam_model.preprocess(X)

        if gt_mask is not None:
            gt_mask = torch.tensor(gt_mask)[...,2] / 255.

            x,y = torch.where(gt_mask == 1)
            bbox = np.array([[y.min(), x.min(), y.max(), x.max()]])
            bbox = self.transform.apply_boxes(bbox, original_size)
            bbox_tensor = torch.as_tensor(bbox, dtype=torch.float, device=self.device)
            gt_mask = gt_mask.to(self.device)
        elif self.avg_bbox is not None:
            if abs(original_size[0] - self.avg_bbox[1]) > 10 or abs(original_size[1] - self.avg_bbox[0]) > 10:
                self.resize_bbox(original_size[::-1])
            bbox = self.transform.apply_boxes(self.avg_bbox[2:], original_size)
            bbox_tensor = torch.as_tensor(bbox, dtype=torch.float, device=self.device)
        else:
            bbox_tensor = None
        
        # model
        with torch.no_grad():
            image_embedding = self.sam_model.image_encoder(X)
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None, boxes=bbox_tensor, masks=None
            )
        
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = self.sam_model.postprocess_masks(
            low_res_masks, input_size, original_size
        )
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        return gt_mask, binary_mask

