import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_video
from torchvision import transforms

class RWTHPhoenixDataset(Dataset):
    """
    RWTH-PHOENIX14T Dataset Loader (DGS)
    Returns:
        visual_features: Tensor [T, 3, 224, 224]
        signer_onehot: Tensor [num_signers]
        language_onehot: Tensor [3] (DGS-German = [0,1,0])
        gloss_targets: Tensor [T_g]
        text_targets: Tensor [T_t]
    """
    def __init__(self, data_root, split="train"):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_signers = 9
        self.language_onehot = torch.tensor([0.0, 1.0, 0.0])
        self.samples = self._load_metadata()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_metadata(self):
        meta_file = os.path.join(self.data_root, f"{self.split}_meta.csv")
        samples = []
        with open(meta_file, 'r') as f:
            for line in f.readlines()[1:]:
                sample_id, video_path, gloss_path, text_path, signer_id = line.strip().split(',')
                samples.append({
                    'video_path': os.path.join(self.data_root, video_path),
                    'gloss_path': os.path.join(self.data_root, gloss_path),
                    'text_path': os.path.join(self.data_root, text_path),
                    'signer_id': int(signer_id)
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video, _, _ = read_video(sample['video_path'], pts_unit='sec')
        video = video.permute(0, 3, 1, 2).float() / 255.0
        video = torch.stack([self.transform(frame) for frame in video])

        gloss_indices = torch.from_numpy(np.load(sample['gloss_path'])).long()
        text_indices = torch.from_numpy(np.load(sample['text_path'])).long()

        signer_onehot = torch.zeros(self.num_signers)
        signer_onehot[sample['signer_id']] = 1.0

        return {
            'visual': video,
            'signer_onehot': signer_onehot,
            'language_onehot': self.language_onehot,
            'gloss': gloss_indices,
            'text': text_indices
        }
