import torch
from torch.utils.data import Dataset

from apedia.data_processing.create_tumor_roi_for_aligned_he import get_ometiff_tissue_mask
from apedia.data_processing.extract_and_visualize_patch_from_wsi import find_all_tissue_patch_locations, get_wsi_patch_ome


class WSIOmeDataset3channelSimple(Dataset):
    """Create dataset from data frame"""

    def __init__(self, wsi, transform=None):
        tissue_mask = get_ometiff_tissue_mask(wsi, staining_name='grayscale')
        pdl1_tissue_locations = find_all_tissue_patch_locations(tissue_mask, coverage=0.6, sensitivity=4)
        self.locations = pdl1_tissue_locations
        self.wsi = wsi
        self.transform = transform

        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = get_wsi_patch_ome(self.locations[idx], None, self.wsi)
        
        target = [self.locations[idx]]
        
        if self.transform:
            sample = self.transform(image=sample)
            result = sample['image']
            return result, target
        
        return sample, target
    
    def get_patch(self, idx):
        sample = get_wsi_patch_ome(self.locations[idx], None, self.wsi)
        sample, self.locations[idx]
        return sample, self.locations[idx]
