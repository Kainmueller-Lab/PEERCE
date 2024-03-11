from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch
# from PIL import Image
import albumentations as A
# from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt




# transform the above function into albumentations transforms:
def get_more_augmented_train_transform_6chan_alb(IMAGE_SIZE, pretrained=True, n_imgs=1, mult=1):
    train_tranforms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1*mult, 0.1*mult), scale=(0.9+(0.1-0.1*mult), 1.1+(-0.1+0.1*mult)), shear=(-20*mult, 20*mult)),
        A.ColorJitter(brightness=0.2*mult, contrast=0.2*mult, hue=0.1*mult, saturation=0.2*mult),
        alb_normalize_transform(pretrained, n_imgs=n_imgs),
        ToTensorV2()
    ], additional_targets={'image0': 'image'})
    return train_tranforms


def get_more_augmented_alsoelastic_train_transform_6chan_alb(IMAGE_SIZE, pretrained=True, n_imgs=1, mult=1):
    train_tranforms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1*mult, 0.1*mult), scale=(0.9+(0.1-0.1*mult), 1.1+(-0.1+0.1*mult)), shear=(-20*mult, 20*mult)),
        A.ColorJitter(brightness=0.2*mult, contrast=0.2*mult, hue=0.1*mult, saturation=0.2*mult),
        A.ElasticTransform(p=1.0, alpha=120*mult, sigma=120*0.05*mult, alpha_affine=120*0.03),
        alb_normalize_transform(pretrained, n_imgs=n_imgs),
        ToTensorV2()
    ], additional_targets={'image0': 'image'})
    return train_tranforms

def get_valid_transform_6chan_alb(IMAGE_SIZE, pretrained=True, n_imgs=1):
    valid_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        alb_normalize_transform(pretrained, n_imgs=n_imgs),
        ToTensorV2()
    ], additional_targets={'image0': 'image'})
    return valid_transform


def alb_normalize_transform(pretrained, n_imgs=1):
    # updated to also normalize tensors with >3 channels
    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(n_imgs-1):
            mean.extend(mean)
            std.extend(std)
        normalize = A.Normalize(
            mean=mean,
            std=std
            )
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for i in range(n_imgs-1):
            mean.extend(mean)
            std.extend(std)
        normalize = A.Normalize(
            mean=mean,
            std=std
        )
        
    return normalize



class WSIPatchDfDataset6channelInMemory(Dataset):
    """Create dataset from data frame"""

    def __init__(self, patch_df, classes=['non_tumor', 'tumor'], cv_split=0, transform=None, 
                 columns=['path_patch_he', 'path_patch_pdl1'], valid=False, pil_image=False, use_masks=False, mask_col='path_patch_mask', shared_cache_img=None, shared_cache_mask=None):
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.transform = transform
        self.patch_df = patch_df
        self.pil_image = pil_image
        self.data_dict_img = shared_cache_img
        self.data_dict_mask = shared_cache_mask
        if valid:
            self.limited_df = patch_df[patch_df.cv_split == cv_split].reset_index(drop=True)
        else:
            self.limited_df = patch_df[patch_df.cv_split != cv_split].reset_index(drop=True)
        self.imgs = [self.limited_df[column].values for column in columns]
        self.targets = self.limited_df.tumors.values
        self.use_masks = use_masks
        if self.use_masks:
            self.masks = self.limited_df[mask_col].values
        
    def __len__(self):
        return len(self.limited_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx not in self.data_dict_img:
            sample = [io.imread(img[idx]) for img in self.imgs]
            sample = np.concatenate(sample, axis=2)
            self.data_dict_img[idx] = sample
        else:
            sample = self.data_dict_img[idx]

        if self.use_masks:
            if idx not in self.data_dict_mask:
                mask = np.load(self.masks[idx]).astype(int)
                self.data_dict_mask[idx] = mask
            else:
                mask = self.data_dict_mask[idx]

        target = self.targets[idx]
        
        if self.transform:
            if self.use_masks:
                if sample.shape[2] == 3:
                    sample = self.transform(image=sample, mask=mask)
                    result = sample['image']
                else:
                    sample = self.transform(image=sample[:, :, :3], image0=sample[:, :, 3:], mask=mask)
                    result = torch.cat((sample['image'], sample['image0']), dim=0)
                return result, target, sample['mask']
            else:
                if sample.shape[2] == 3:
                    sample = self.transform(image=sample)
                    sample = sample['image']
                else:
                    sample = self.transform(image=sample[:, :, :3], image0=sample[:, :, 3:])
                    sample = torch.cat((sample['image'], sample['image0']), dim=0)
                return sample, target
        
        return sample, target



class WSIPatchDfDataset6channel(Dataset):
    """Create dataset from data frame"""

    def __init__(self, patch_df, classes=['non_tumor', 'tumor'], cv_split=0, transform=None, 
                 columns=['path_patch_he', 'path_patch_pdl1'], valid=False, pil_image=False, use_masks=False, mask_col='path_patch_mask',
                 angio='all'):
        # options for angio are:
        # 'all' - use all patches (regardless of verified angio status)
        # 'verified' - use only patches with verified angio status
        # 'train_all_valid_verified' - use only verified patches for testing, and all other cv_split + non-verified angio patches for training
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.transform = transform
        self.patch_df = patch_df
        self.pil_image = pil_image
        if angio == 'all':
            if valid:
                self.limited_df = patch_df[patch_df.cv_split == cv_split].reset_index(drop=True)
            else:
                self.limited_df = patch_df[patch_df.cv_split != cv_split].reset_index(drop=True)
        elif angio == 'verified':
            if valid:
                self.limited_df = patch_df[(patch_df.cv_split == cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
            else:
                self.limited_df = patch_df[(patch_df.cv_split != cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
        elif angio == 'train_all_valid_verified':
            if valid:
                self.limited_df = patch_df[(patch_df.cv_split == cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
            else:
                self.limited_df = patch_df[(patch_df.cv_split != cv_split) | ~patch_df.verified_angiosarcoma].reset_index(drop=True)
        self.imgs = [self.limited_df[column].values for column in columns]
        self.targets = self.limited_df.tumors.values
        self.use_masks = use_masks
        if self.use_masks:
            self.masks = self.limited_df[mask_col].values
        
    def __len__(self):
        return len(self.limited_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [io.imread(img[idx]) for img in self.imgs]
        # if len(sample) > 1: # this check isn't needed.
        sample = np.concatenate(sample, axis=2)
        
        if self.use_masks:
            mask = np.load(self.masks[idx])
            mask = mask.astype(int)
        # if self.pil_image:
        #     sample = Image.fromarray(sample)
        # else:
        #     sample = F.to_tensor(sample)
        
        target = self.targets[idx]
        
        if self.transform:
            if self.use_masks:
                if sample.shape[2] == 3:
                    sample = self.transform(image=sample, mask=mask)
                    result = sample['image']
                else:
                    sample = self.transform(image=sample[:, :, :3], image0=sample[:, :, 3:], mask=mask)
                    result = torch.concat((sample['image'], sample['image0']), dim=0)
                return result, target, sample['mask']
            else:
                if sample.shape[2] == 3:
                    sample = self.transform(image=sample)
                    sample = sample['image']
                else:
                    sample = self.transform(image=sample[:, :, :3], image0=sample[:, :, 3:])
                    sample = torch.concat((sample['image'], sample['image0']), dim=0)
                return sample, target
        
        return sample, target
    

def viz_dset_output(dataset, idx):
    x, y = dataset[idx]
    imgs = np.transpose(x.numpy(), (1,2,0))
    plt.imshow(imgs[:,:,:3])
    plt.show()
    plt.imshow(imgs[:,:, 3:])
    plt.show()
    
    
# if __name__ == '__main__':
#     import pandas as pd
#     from datasets import get_more_augmented_train_transform_6chan, get_valid_transform_6chan #, normalize_transform
#     path_df_patches_tumor_non_tumor = '/home/fabian/raid5/Angiosarkom-Scans/pdl1_he_mask_patches_9jan23_tumor_non_tumor_cv_splits.feather'
#     df = pd.read_feather(path_df_patches_tumor_non_tumor)
#     dataset_train = WSIPatchDfDataset6channel(df, valid=False, transform=get_more_augmented_train_transform_6chan_alb(512), columns=['path_patch_he'],
#                                               use_masks=True)
#     dataset_valid = WSIPatchDfDataset6channel(df, valid=True, transform=get_valid_transform_6chan_alb(512))
#     # x, y = dataset_valid[10]
#     x, y, m = dataset_train[10]
#     print('done')
    