import os

import imageio
from torch.utils.data import Dataset, DataLoader

class SegDataset(Dataset):
    def __init__(self, base_dir, mode):
        
        self.data_dir = os.path.join(base_dir, 'image')
        self.mask_dir = os.path.join(base_dir, 'gt_image') if mode == 'train' else None

    def __len__(self):
        return len(os.listdir(self.data_dir))


    def __getitem__(self, idx):
        
        file = os.listdir(self.data_dir)[idx]
        x = imageio.imread(os.path.join(self.data_dir, file))
        m_file = file.split('_')
        m_file.insert(1, 'road')

        if self.mask_dir:
            mask = imageio.imread(os.path.join(self.mask_dir, '_'.join(m_file)))
        else:
            mask = None

        return x, mask


def trivial_collate(batch):
    return batch[0]

def get_dataloader(base_data_dir, mode):

    dataset = SegDataset(base_data_dir, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=8,
        collate_fn=trivial_collate
    )
    return dataloader