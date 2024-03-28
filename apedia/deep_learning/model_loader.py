import os
import requests
import torch
from tqdm import tqdm
from tempfile import NamedTemporaryFile

from apedia.deep_learning.unet_functionality import NInputChanUnet


def download_file_from_url(url, destination_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    
    block_size = 1024  # 1 Kibibyte for reading
    progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, unit_divisor=1024)
    
    if response.status_code == 200:
        # Use a temporary file to download
        with NamedTemporaryFile(delete=False) as temp_file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                temp_file.write(data)
            temp_path = temp_file.name  # Store temporary file name to move it later

        progress_bar.close()
        
        # Verify download completeness
        downloaded_size = os.path.getsize(temp_path)
        if total_size_in_bytes != 0 and downloaded_size == total_size_in_bytes:
            # Ensure the destination directory exists
            destination_dir = os.path.dirname(destination_path)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir, exist_ok=True)
            # Move temporary file to destination if fully downloaded
            os.rename(temp_path, destination_path)
            print(f"Downloaded weights to {destination_path}")
        else:
            # Delete temporary file if download is incomplete
            os.remove(temp_path)
            print("ERROR, something went wrong. Download was not complete.")
    else:
        progress_bar.close()
        raise Exception(f"Failed to download file from {url}")


def get_tp_pred_model():
    # Define the path where the weights should be
    current_dir = os.path.dirname(__file__)
    path_network_weights = os.path.join(current_dir, '..', '..', 'weights', 'tp_pred_model_checkpoint.pth')
    
    # Check if weights exist
    if not os.path.exists(path_network_weights):
        print("Model weights not found locally. Downloading from Hugging Face...")
        # If weights don't exist, download them
        url = "https://huggingface.co/FabianReith/apedia/resolve/main/tp_pred_model_checkpoint.pth"
        # Make sure the directory exists
        os.makedirs(os.path.dirname(path_network_weights), exist_ok=True)
        download_file_from_url(url, path_network_weights)
    
    encoder_name = 'tu-tf_efficientnetv2_m.in21k_ft_in1k'
    tp_pred_model = NInputChanUnet(n_channels_in=3, model_cls=None, encoder_name=encoder_name, encoder_weights=None, 
                                    in_channels=3, classes=2, activation=None)
    tp_pred_model = tp_pred_model.to('cuda')
    tp_pred_model.eval()

    # Load the weights
    tp_pred_model_checkpoint = torch.load(path_network_weights, map_location='cuda')
    load_status = tp_pred_model.load_state_dict(tp_pred_model_checkpoint['model_state_dict'])
    print(f"Model loaded with status: {load_status}")
    
    return tp_pred_model


def get_seg_model():
    # Define the path where the segmentation model weights should be stored
    current_dir = os.path.dirname(__file__)
    path_network_weights = os.path.join(current_dir, '..', '..', 'weights', 'seg_model_checkpoint.pth')
    
    # Check if weights exist locally
    if not os.path.exists(path_network_weights):
        print("Segmentation model weights not found locally. Downloading from Hugging Face...")
        # If weights don't exist locally, download them
        url = "https://huggingface.co/FabianReith/apedia/resolve/main/seg_model_checkpoint.pth"
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path_network_weights), exist_ok=True)
        download_file_from_url(url, path_network_weights)
    
    # Initialize the segmentation model
    seg_model = NInputChanUnet(n_channels_in=3, model_cls=None, encoder_name="timm-efficientnet-b5", encoder_weights=None, 
                               in_channels=3, classes=4, activation=None)
    seg_model = seg_model.to('cuda')
    seg_model.eval()

    # Load the weights
    seg_model_checkpoint = torch.load(path_network_weights, map_location='cuda')
    load_status = seg_model.load_state_dict(seg_model_checkpoint['model_state_dict'])
    print(f"Segmentation model loaded with status: {load_status}")
    
    return seg_model
