from datetime import datetime
from pathlib import Path
from cellpose import models as cp_models
# import pandas as pd
# import torch

from apedia.data_processing.deepliif_hema_patch import MakeHemaPatch
from apedia.data_processing.model_loader import get_seg_model, get_tp_pred_model
from apedia.data_processing.timm_convnext_model_ds_predictions_viz import WSIOmeDataset3channelSimple
from apedia.data_processing.wsi_patch_df_6channel_dataset import get_valid_transform_6chan_alb
# from apedia.deep_learning.unet_functionality import NInputChanUnet
from apedia.utils.read_and_visualize_ome_tiff import OmeTiffFile
from apedia.deep_learning.wsi_tps_prediction import TpsSegmentationDataset, identify_tumor_patches_wsi, predict_cell_types_wsi


def predict_tps(ometiff_path, output_folder, tp_pred_model=None, seg_model=None, cellpose_model=None):    
    # Define the output folder
    wsi_name = Path(ometiff_path).name
    wsi_name_id = wsi_name.split('.')[0]
    date_string = datetime.now().strftime("%d%b%y")
    wsi_prediction_folder = Path(output_folder) / f"{wsi_name}_{date_string}"
    wsi_prediction_folder.mkdir(parents=True, exist_ok=True)
    
    # Load the ometiff WSI, create the dataset
    wsi = OmeTiffFile(ometiff_path)
    ds_valid_transform = get_valid_transform_6chan_alb(512)
    wsi_dataset = WSIOmeDataset3channelSimple(wsi, transform=ds_valid_transform)
    
    # Create tumor patch prediction model
    if tp_pred_model is None:
        tp_pred_model = get_tp_pred_model()
    
    # Identify tumor patches in the WSI
    print('Identify all tumor patches within the WSI')
    wsi_prediction_df = identify_tumor_patches_wsi(wsi_dataset, tp_pred_model, wsi_name_id, wsi_prediction_folder, threshold=0.6, viz=False)
    filtered_df = wsi_prediction_df[wsi_prediction_df['prediction'] > 0.6].copy().reset_index(drop=True)
    
    if len(filtered_df) < 10:
        wsi_prediction_df = identify_tumor_patches_wsi(wsi_dataset, tp_pred_model, wsi_name_id, wsi_prediction_folder, threshold=0.0, viz=False)
        filtered_df = wsi_prediction_df.sort_values('prediction', ascending=False).head(10).copy().reset_index(drop=True)
    
    # TODO - remove this
    # for debugging, reduce filtered_df to 20 random samples
    filtered_df = filtered_df.sample(20).copy().reset_index(drop=True)
    
    # Predict cell types in the tumor patches
    # Create segmentation model
    if seg_model is None:
        seg_model = get_seg_model()
        
    # Create Cellpose model
    if cellpose_model is None:
        cellpose_model = cp_models.Cellpose(gpu=True, model_type='nuclei')
    
    print('Predict cell types in the tumor patches')
    dset_tps = TpsSegmentationDataset(filtered_df)
    make_hema = MakeHemaPatch()
    filtered_df = predict_cell_types_wsi(dset_tps, cellpose_model, make_hema, seg_model, filtered_df, wsi_prediction_folder, wsi_name_id)
    


if __name__ == '__main__':
    ometiff_path = '/home/fabian/raid5/Angiosarkom-Scans/ometiff_PD-L1/W134-22_PD-L1.ome.tiff'
    output_folder = "/home/fabian/projects/phd/APEDIA/data/outputs/wsi_tps_predictions/"
    
    predict_tps(ometiff_path, output_folder)