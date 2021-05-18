import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import h5py
import random

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

class TestSet(Dataset):
    def __init__(self, args, input_transform=None, ref_transform=None):
        super(TestSet, self).__init__()

        self.upsample_factor = args.upsample_factor

        image_h5_file = h5py.File(args.image_dataset_dir, 'r')
        ref_h5_file = h5py.File(args.ref_dataset_dir, 'r')
        image_dataset = image_h5_file['data']
        ref_dataset = ref_h5_file['data']

        self.image_datasets = image_dataset
        self.ref_datasets = ref_dataset
        self.total_count = image_dataset.shape[0]

        self.input_transform = input_transform
        self.ref_transform = ref_transform

    def __len__(self):
        return self.image_datasets.shape[0]

    def __getitem__(self, index):
        hr_height, hr_width = self.ref_datasets.shape[2], self.ref_datasets.shape[3]

        lr = self.image_datasets[index, 1:, :, :]  # 5 LR_MC frames [1:5]
        lr_up = np.apply_along_axis(lambda x: np.array(Image.fromarray(x).resize(
            (hr_width, hr_height), Image.BICUBIC), axis=1, arr=lr))  # 5 LR_Bic_MC frames [1:5]
        hr = self.ref_datasets[index, [4], :, :]  # HR center frame
        ref = self.ref_datasets[index, [0], :, :]  # HR first frame
        ref_down = np.apply_along_axis(lambda x: np.array(Image.fromarray(x).resize(
            (hr_width//self.upsample_factor, hr_height//self.upsample_factor), Image.BICUBIC)), axis=1, arr=ref)  # [0] #TODO change if uf is different
        ref_dup = np.apply_along_axis(lambda x: np.array(Image.fromarray(x).resize(
            (hr_width, hr_height), Image.BICUBIC)), axis=1, arr=ref_down)  # [0]
        lr = lr.astype(np.float32)
        ref = ref.astype(np.float32)

        #   Notice that image is the bicubic upscaled LR image patch, in float format, in range [0, 1]
        # lr = lr / 255.0
        #   Notice that target is the HR image patch, in uint8 format, in range [0, 255]
        # ref = ref / 255.0

        lr = torch.from_numpy(lr)
        lr_up = torch.from_numpy(lr_up)
        hr = torch.from_numpy(hr)
        ref = torch.from_numpy(ref)
        ref_dup = torch.from_numpy(ref_dup)

        sample = {'LR': lr,  # LR_MC
                  'LR_sr': lr_up,  # LR_Bic_MC
                  'HR': hr,  # HR
                  'Ref': ref,  # HR
                  'Ref_sr': ref_dup}  # HR_Bic

        if self.transform:
            sample = self.transform(sample)
        return sample

class TrainSet(Dataset):
    # LR original = LR_MC
    # reference_dataset = HR
    # reference up/down --> calculate
    # LR upsampled = LR_Bic_MC
    def __init__(self, args, input_transform=None, ref_transform=None):
        super(TrainSet, self).__init__()

        self.upsample_factor = args.upsample_factor

        image_h5_file = h5py.File(args.image_dataset_dir, 'r') #dataset of individual videos
        ref_h5_file = h5py.File(args.ref_dataset_dir, 'r')
        image_dataset = image_h5_file['data']
        print("img: ", image_dataset.shape)
        die
        ref_dataset = ref_h5_file['data']

        self.image_datasets = image_dataset
        self.ref_datasets = ref_dataset
        self.total_count = image_dataset.shape[0]

        self.input_transform = input_transform
        self.ref_transform = ref_transform

    def __len__(self):
        return self.image_datasets.shape[0]

    def __getitem__(self, index):
        # [total_indices, 6_frames, width, height]
        hr_height, hr_width = self.ref_datasets.shape[2], self.ref_datasets.shape[3]

        # lr = lr.astype(np.float32)
        lr = self.image_datasets[index:index+1, 1:, :, :]  # 5 LR_MC frames [1:5]
        lr = torch.from_numpy(lr)

        # LR Scaled up (x2)
        lr_up = F.interpolate(lr, scale_factor=2, mode="bicubic") # 5 LR_Bic_MC frames [1:5]  
        lr = F.interpolate(lr, scale_factor=0.5, mode="bicubic")

        hr = self.ref_datasets[index:index+1, [4], :, :]  # HR center frame
        hr = hr.astype(np.float32)
        hr = torch.from_numpy(hr)

        # Ref frame
        ref = self.ref_datasets[index:index+1, [0], :, :]  # HR first frame
        ref = ref.astype(np.float32)
        ref = torch.from_numpy(ref)
        # ref downsample
        ref_down = F.interpolate(ref, scale_factor=0.25, mode="bicubic")  # [0]
        # ref down + upsample
        ref_dup = F.interpolate(ref_down, scale_factor=4, mode="bicubic") # [0]

        #   Notice that image is the bicubic upscaled LR image patch, in float format, in range [0, 1]
        # lr = lr / 255.0
        #   Notice that target is the HR image patch, in uint8 format, in range [0, 255]
        # ref = ref / 255.0

        sample = {'LR': lr.squeeze(0),  # LR_MC [5]
                  'LR_sr': lr_up.squeeze(0),  # LR_Bic_MC [5]
                  'HR': hr.squeeze(0),  # HR [1]
                  'Ref': ref.squeeze(0),  # HR [1]
                  'Ref_sr': ref_dup.squeeze(0)}  # HR_Bic [1]

        # if self.transform:
        #     sample = self.transform(sample)
        return sample



    # def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
    #     self.input_list = sorted(glob.glob(os.path.join(
    #         args.dataset_dir, 'test/CUFED5', '*_0.png')))
    #     self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5',
    #                                                   '*_' + ref_level + '.png')))
    #     self.transform = transform

    # def __len__(self):
    #     return len(self.input_list)

    # def __getitem__(self, idx):
    #     # HR
    #     HR = imread(self.input_list[idx])
    #     h, w = HR.shape[:2]
    #     h, w = h//4*4, w//4*4
    #     HR = HR[:h, :w, :]  # crop to the multiple of 4

    #     # LR and LR_sr
    #     LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
    #     LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

    #     # Ref and Ref_sr
    #     Ref = imread(self.ref_list[idx])
    #     h2, w2 = Ref.shape[:2]
    #     h2, w2 = h2//4*4, w2//4*4
    #     Ref = Ref[:h2, :w2, :]
    #     Ref_sr = np.array(Image.fromarray(Ref).resize(
    #         (w2//4, h2//4), Image.BICUBIC))
    #     Ref_sr = np.array(Image.fromarray(
    #         Ref_sr).resize((w2, h2), Image.BICUBIC))

    #     # change type
    #     LR = LR.astype(np.float32)
    #     LR_sr = LR_sr.astype(np.float32)
    #     HR = HR.astype(np.float32)
    #     Ref = Ref.astype(np.float32)
    #     Ref_sr = Ref_sr.astype(np.float32)

    #     # rgb range to [-1, 1]
    #     LR = LR / 127.5 - 1.
    #     LR_sr = LR_sr / 127.5 - 1.
    #     HR = HR / 127.5 - 1.
    #     Ref = Ref / 127.5 - 1.
    #     Ref_sr = Ref_sr / 127.5 - 1.

    #     sample = {'LR': LR,
    #               'LR_sr': LR_sr,
    #               'HR': HR,
    #               'Ref': Ref,
    #               'Ref_sr': Ref_sr}

    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample


# class TrainSet(Dataset):
#     def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
#         self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in
#                                   os.listdir(os.path.join(args.dataset_dir, 'train/input'))])
#         self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in
#                                 os.listdir(os.path.join(args.dataset_dir, 'train/ref'))])
#         self.transform = transform

#     def __len__(self):
#         return len(self.input_list)

#     def __getitem__(self, idx):
#         # HR
#         HR = imread(self.input_list[idx])
#         h, w = HR.shape[:2]
#         # HR = HR[:h//4*4, :w//4*4, :]

#         # LR and LR_sr
#         LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
#         LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

#         # Ref and Ref_sr
#         Ref_sub = imread(self.ref_list[idx])
#         h2, w2 = Ref_sub.shape[:2]
#         Ref_sr_sub = np.array(Image.fromarray(
#             Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
#         Ref_sr_sub = np.array(Image.fromarray(
#             Ref_sr_sub).resize((w2, h2), Image.BICUBIC))

#         # complete ref and ref_sr to the same size, to use batch_size > 1
#         Ref = np.zeros((160, 160, 3))
#         Ref_sr = np.zeros((160, 160, 3))
#         Ref[:h2, :w2, :] = Ref_sub
#         Ref_sr[:h2, :w2, :] = Ref_sr_sub

#         # change type
#         LR = LR.astype(np.float32)
#         LR_sr = LR_sr.astype(np.float32)
#         HR = HR.astype(np.float32)
#         Ref = Ref.astype(np.float32)
#         Ref_sr = Ref_sr.astype(np.float32)

#         # rgb range to [-1, 1]
#         LR = LR / 127.5 - 1.
#         LR_sr = LR_sr / 127.5 - 1.
#         HR = HR / 127.5 - 1.
#         Ref = Ref / 127.5 - 1.
#         Ref_sr = Ref_sr / 127.5 - 1.

#         sample = {'LR': LR,  # LR_MC
#                   'LR_sr': LR_sr,
#                   'HR': HR,  # HR
#                   'Ref': Ref,  # LR_Bic_MC
#                   'Ref_sr': Ref_sr}

#         if self.transform:
#             sample = self.transform(sample)
#         return sample