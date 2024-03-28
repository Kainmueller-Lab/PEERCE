import argparse
import datetime
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from apedia.deep_learning.unet_functionality import NInputChanUnet
from apedia.data_processing.datasets import get_data_loaders
from apedia.data_processing.wsi_patch_df_6channel_dataset import get_more_augmented_train_transform_6chan_alb, get_valid_transform_6chan_alb, get_more_augmented_alsoelastic_train_transform_6chan_alb
from apedia.utils.params import train_cell_type_detection_params
from apedia.deep_learning.dl_utils import save_model
from apedia.utils.analyze_segmentation_output import plot_circles_and_roi_points, visualize_image_and_model_output
from apedia.deep_learning.train_unet_cell_segmentation_and_valid import train_unet_cell_segmentation, SegmentationDfDataset


def train_cell_type_detector(params):
    # Load the dataframe
    df = pd.read_feather(params['df_path'])    
    # ensure that output dir is a Path object
    params['output_dir'] = Path(params['output_dir'])
    
    # Convert class_weights from list of floats to torch.Tensor if provided
    if params['class_weights'] is not None:
        params['class_weights'] = torch.tensor(params['class_weights'], dtype=torch.float32)
    
    if params['do_elastic']:
        dset_train = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'],
                                           patch_col=params['patch_col'], instance_seg_col=params['instance_seg_col'],
                                           transform=get_more_augmented_alsoelastic_train_transform_6chan_alb(512, mult=params['aug_mult']),
                                           valid=False, angio='all')
    else:
        dset_train = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'],
                                           patch_col=params['patch_col'], instance_seg_col=params['instance_seg_col'],
                                           transform=get_more_augmented_train_transform_6chan_alb(512, mult=params['aug_mult']),
                                           valid=False, angio='all')
                                            
    dset_val = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'],
                                     patch_col=params['patch_col'], instance_seg_col=params['instance_seg_col'],
                                     transform=get_valid_transform_6chan_alb(512), valid=True, angio='all')

    dset_val_notransform = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'],
                                                 patch_col=params['patch_col'], instance_seg_col=params['instance_seg_col'],
                                    transform=None, valid=True, angio='all')
    train_loader, valid_loader = get_data_loaders(dset_train, dset_val, batch_size=params['bs'], num_workers=params['num_workers'])

    model = NInputChanUnet(n_channels_in=3, model_cls=None, encoder_name=params['encoder_name'], encoder_weights="imagenet", 
                        in_channels=3, classes=params['num_classes'], activation=None).to(params['device'])

    criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'], weight=params['class_weights'].to(params['device']))
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])

    loss_acc_dict = None
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch+1} of {params['epochs']}")
        loss_acc_dict = train_unet_cell_segmentation(model, train_loader, valid_loader, optimizer, criterion, loss_acc_dict=loss_acc_dict, disable_tqdm=params['disable_tqdm'])
        scheduler.step()

    if not params['do_not_analyze']:
        current_date_string = datetime.datetime.now().strftime('%d%b%y').lower()
        path_analysis_output = Path(params['output_dir']) / f'cell_type_detection_analysis_output_{current_date_string}'
        path_analysis_output.mkdir(parents=True, exist_ok=True)
        path_analysis_output_cv = path_analysis_output / f"cv_split_{params['cv_split']}"
        path_analysis_output_cv.mkdir(exist_ok=True)
        
        with open(path_analysis_output_cv / 'loss_acc_dict.pkl', 'wb') as f:
            pickle.dump(loss_acc_dict, f)
        
        save_model(params['epochs'], model, 'NInputChanUnet', optimizer, criterion, path_analysis_output_cv, f"{current_date_string}_NInputChanUnet_cv_split{params['cv_split']}_epochs{params['epochs']}")

        for idx in tqdm(range(len(dset_val))):
            model.eval()
            image_model_input, mask, inst_seg = dset_val[idx]
            true_img, _, _ = dset_val_notransform[idx]
            model_out = model(image_model_input.unsqueeze(0).to(params['device']))
            model_out = model_out.cpu().detach().numpy().squeeze()
            
            info_ground_truth = f"Unet cell segmentation ground truth for patch {Path(dset_val.patches[idx]).stem}"
            out_ground_truth_name = f"{Path(dset_val.patches[idx]).stem}_cell_annotations.jpg"
            info_prediction = f"Unet cell segmentation prediction for patch {Path(dset_val.patches[idx]).stem}"
            out_prediction_name = f"{Path(dset_val.patches[idx]).stem}_segmentation_prediction_analysis.jpg"
            
            plot_circles_and_roi_points(true_img, inst_seg, info_ground_truth, path_analysis_output_cv / out_ground_truth_name)
            visualize_image_and_model_output(true_img, model_out, inst_seg, info_prediction, path_analysis_output_cv / out_prediction_name)


def main():
    parser = argparse.ArgumentParser(description='Train Cell Type Detector')
    # Mandatory parameters
    parser.add_argument('--df_path', type=str, required=False, help='Path to the dataframe containing the dataset information')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs')
    
    # Optional parameters with defaults from train_cell_type_detection_params
    parser.add_argument('--cv_split', type=int, help='Cross-validation split index', default=train_cell_type_detection_params['cv_split'])
    parser.add_argument('--mask_col', type=str, help='Column name for mask paths', default=train_cell_type_detection_params['mask_col'])
    parser.add_argument('--patch_col', type=str, help='Column name for patch paths', default=train_cell_type_detection_params['patch_col'])
    parser.add_argument('--instance_seg_col', type=str, help='Column name for instance segmentation paths', default=train_cell_type_detection_params['instance_seg_col'])
    parser.add_argument('--bs', type=int, help='Batch size', default=train_cell_type_detection_params['bs'])
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=train_cell_type_detection_params['epochs'])
    parser.add_argument('--lr', type=float, help='Learning rate', default=train_cell_type_detection_params['lr'])
    parser.add_argument('--aug_mult', type=float, help='Augmentation multiplier', default=train_cell_type_detection_params['aug_mult'])
    parser.add_argument('--encoder_name', type=str, help='Encoder name for the UNet', default=train_cell_type_detection_params['encoder_name'])
    parser.add_argument('--num_input_channels', type=int, help='Number of input channels for the model', default=train_cell_type_detection_params['num_input_channels'])
    parser.add_argument('--num_classes', type=int, help='Number of output classes for the model', default=train_cell_type_detection_params['num_classes'])
    parser.add_argument('--num_workers', type=int, help='Number of workers for the data loaders', default=train_cell_type_detection_params['num_workers'])
    parser.add_argument('--device', type=str, help='Device to train on ("cuda" or "cpu")', default=train_cell_type_detection_params['device'])
    parser.add_argument('--do_elastic', action='store_true', help='Use elastic transformations', default=train_cell_type_detection_params['do_elastic'])
    parser.add_argument('--do_not_analyze', action='store_true', help='Skip analysis after training', default=train_cell_type_detection_params['do_not_analyze'])
    parser.add_argument('--label_smoothing', type=float, help='Label smoothing value', default=train_cell_type_detection_params['label_smoothing'])
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer', default=train_cell_type_detection_params['weight_decay'])
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable TQDM progress bars', default=train_cell_type_detection_params['disable_tqdm'])
    parser.add_argument('--class_weights', type=float, nargs='+', help='Class weights for loss calculation', default=train_cell_type_detection_params['class_weights'])

    args = parser.parse_args()

    # Update params dict with args
    params = vars(args)

    # for now, fixed df_path and output_dir
    params['df_path'] = "/home/fabian/projects/phd/APEDIA/data/example_df_segmentation.feather"
    # example, preprocessed via APEDIA:
    params['df_path'] = '/home/fabian/projects/phd/APEDIA/data/example_df_segmentation_reduced.feather'
    
    params['output_dir'] = '/home/fabian/projects/phd/APEDIA/data/outputs'
    params['epochs'] = 2

    # Call the training function
    train_cell_type_detector(params)


if __name__ == '__main__':
    main()
