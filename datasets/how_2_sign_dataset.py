import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_video
from torchvision import transforms

class How2SignDataset(Dataset):
    """
    How2Sign Dataset Loader (ASL)
    Returns:
        visual_features: Tensor [T, 3, 224, 224]
        signer_onehot: Tensor [num_signers]
        language_onehot: Tensor [3] (ASL-English = [1,0,0])
        gloss_targets: Tensor [T_g]
        text_targets: Tensor [T_t]
    """
    def __init__(self, data_root, split="train"):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_signers = 11
        self.language_onehot = torch.tensor([1.0, 0.0, 0.0])
        self.samples = self._load_metadata()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_metadata(self):
        # Expecting a metadata file mapping sample_id → video_path, gloss_path, text_path, signer_id
        meta_file = os.path.join(self.data_root, f"{self.split}_meta.csv")
        samples = []
        with open(meta_file, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
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

        # Load video frames (torchvision read_video returns (T, H, W, C))
        video, _, _ = read_video(sample['video_path'], pts_unit='sec')
        video = video.permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
        video = torch.stack([self.transform(frame) for frame in video])  # Resize, Normalize

        # Load gloss sequence (pre-tokenized indices)
        gloss_indices = torch.from_numpy(np.load(sample['gloss_path'])).long()

        # Load text sequence (pre-tokenized indices)
        text_indices = torch.from_numpy(np.load(sample['text_path'])).long()

        # Signer one-hot
        signer_onehot = torch.zeros(self.num_signers)
        signer_onehot[sample['signer_id']] = 1.0

        return {
            'visual': video,  # [T, 3, 224, 224]
            'signer_onehot': signer_onehot,  # [11]
            'language_onehot': self.language_onehot,  # [3]
            'gloss': gloss_indices,  # [T_g]
            'text': text_indices  # [T_t]
        }
