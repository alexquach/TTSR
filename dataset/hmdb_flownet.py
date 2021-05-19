import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import h5py
import random

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2, 0, 1))
        LR_sr = LR_sr.transpose((2, 0, 1))
        HR = HR.transpose((2, 0, 1))
        Ref = Ref.transpose((2, 0, 1))
        Ref_sr = Ref_sr.transpose((2, 0, 1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}

class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        hf = h5py.File(args.hf5_dataset, 'r')
        self.lr_list = torch.tensor(hf.get('lr'))
        self.hr_list = torch.tensor(hf.get('hr'))
                                
        self.transform = transform

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, idx):
        lr_frames = self.lr_list[idx] # [5, 3, 40, 40]
        hr_frames = self.hr_list[idx] # [2, 3, 160, 160]

        LR_sr = transforms.functional.resize(lr_frames, (160, 160))
        HR = hr_frames[0]
        Ref = hr_frames[1]

        Ref_down = transforms.functional.resize(Ref, (40, 40))
        Ref_sr = transforms.functional.resize(Ref_down, (160, 160))
        

        sample = {'LR': lr_frames,  # [5, 3, 40, 40]
                  'LR_sr': LR_sr,   # [5, 3, 160, 160]
                  'HR': HR,         # [1, 3, 160, 160]
                  'Ref': Ref,       # [1, 3, 160, 160]
                  'Ref_sr': Ref_sr} # [1, 3, 160, 160]

        # if self.transform:
        #     sample = self.transform(sample)
        return sample