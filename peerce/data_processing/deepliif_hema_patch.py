from torchvision import transforms
import torch
import numpy as np
import os
from torchvision.transforms import ToPILImage

from peerce.data_processing.deepliif_networks import define_G
from peerce.deep_learning.model_loader import download_file_from_url


def transform_deepliif(img):
    return transforms.Compose([
        # transforms.Lambda(lambda i: __make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(img).unsqueeze(0)
    
    
def inverse_transform_deepliif(tensor, to_pil=True, to_array=True):
    # Reverse the normalization
    inv_normalize = transforms.Normalize(
        mean=[-1.0, -1.0, -1.0],
        std=[2.0, 2.0, 2.0]
    )
    inv_tensor = inv_normalize(tensor.squeeze(0))

    if to_pil:
        # Convert the tensor back to a PIL Image
        inv_img = ToPILImage()(inv_tensor)

        # Convert the image pixel values to the range [0, 255]
        inv_img = inv_img.convert("RGB")
        if to_array:
            return np.array(inv_img)
        else:
            return inv_img
    else:
        return inv_tensor
    
    
class MakeHemaPatch:
    def __init__(self, path_network_weights=None):
        if path_network_weights is None:
            # Construct the path to the weights directory relative to this file
            current_dir = os.path.dirname(__file__)
            path_network_weights = os.path.join(current_dir, '..', 'utils', 'weights', 'deepliif_latest_net_G1.pth')
            
            # Check if weights exist locally
            if not os.path.exists(path_network_weights):
                print("Hema model weights not found locally. Downloading from Hugging Face...")
                # If weights don't exist locally, download them
                url = "https://huggingface.co/FabianReith/apedia/resolve/main/deepliif_latest_net_G1.pth"
                # Ensure the directory exists
                os.makedirs(os.path.dirname(path_network_weights), exist_ok=True)
                download_file_from_url(url, path_network_weights)
        
        hema_generator = define_G(3, 3, 64, 'resnet_9blocks', 'batch', True, 'normal', 0.02, [0], 'zero')
        if isinstance(hema_generator, torch.nn.DataParallel):
            hema_generator = hema_generator.module
        hema_generator.load_state_dict(torch.load(path_network_weights, map_location='cuda'))
        self.hema_generator = hema_generator.cuda()
        
    def __call__(self, img):
        img = transform_deepliif(img)
        img = img.cuda()
        img = self.hema_generator(img)
        img = inverse_transform_deepliif(img)
        return img