from torchvision import transforms
import torch
import numpy as np
import os
from torchvision.transforms import ToPILImage

from apedia.data_processing.deepliif_networks import define_G



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
    def __init__(self, path_network_weights=None): # "/home/fabian/projects/phd/deepliif/DeepLIIF/model-server/DeepLIIF_Latest_Model/latest_net_G1.pth"
        if path_network_weights is None:
            # Construct the path to the weights directory relative to this file
            current_dir = os.path.dirname(__file__)
            path_network_weights = os.path.join(current_dir, '..', '..', 'weights', 'deepliif_latest_net_G1.pth')

        
        hema_generator = define_G(3, 3, 64, 'resnet_9blocks', 'batch', True, 'normal', 0.02, [0], 'zero')
        if isinstance(hema_generator, torch.nn.DataParallel):
            hema_generator = hema_generator.module
        hema_generator.load_state_dict(torch.load(path_network_weights))
        self.hema_generator = hema_generator.cuda()
        
    def __call__(self, img):
        img = transform_deepliif(img)
        img = img.cuda()
        img = self.hema_generator(img)
        img = inverse_transform_deepliif(img)
        return img