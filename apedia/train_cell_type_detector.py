import argparse
import datetime
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from apedia.deep_learning.unet_functionality import NInputChanUnet, train_unet_cell_segmentation
from apedia.data_processing.datasets import get_data_loaders, SegmentationDfDataset
from apedia.data_processing.wsi_patch_df_6channel_dataset import get_more_augmented_train_transform_6chan_alb, get_valid_transform_6chan_alb, get_more_augmented_alsoelastic_train_transform_6chan_alb
# from apedia.utils.params import default_cell_type_detector_params
from apedia.deep_learning.dl_utils import save_model
from apedia.utils.analyze_segmentation_output import plot_circles_and_roi_points, visualize_image_and_model_output


def train_cell_type_detector(params):
    # Load the dataframe
    df = pd.read_feather(params['df_path'])    
    
    if params['do_elastic']:
        dset_train = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'],
                                            transform=get_more_augmented_alsoelastic_train_transform_6chan_alb(512, mult=params['aug_mult']),
                                            valid=False, angio='all')
    else:
        dset_train = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'], 
                                            transform=get_more_augmented_train_transform_6chan_alb(512, mult=params['aug_mult']),
                                            valid=False, angio='all')
                                            
    dset_val = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'], 
                                        transform=get_valid_transform_6chan_alb(512),
                                        valid=True, angio='all')

    dset_val_notransform = SegmentationDfDataset(df, cv_split=params['cv_split'], mask_col=params['mask_col'], 
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

    if params['do_analyze']:
        current_date_string = datetime.datetime.now().strftime('%d%b%y').lower()
        path_analysis_output = Path(params['output_dir']) / f'analysis_output_{current_date_string}'
        path_analysis_output.mkdir(parents=True, exist_ok=True)
        path_analysis_output_cv = path_analysis_output / f"cv_split_{params['cv_split']}"
        path_analysis_output_cv.mkdir(exist_ok=True)
        
        with open(path_analysis_output_cv / 'loss_acc_dict.pkl', 'wb') as f:
            pickle.dump(loss_acc_dict, f)
        
        save_model(params['epochs'], model, 'NInputChanUnet', optimizer, criterion, path_analysis_output_cv, f"{params['current_date_string']}_NInputChanUnet_cv_split{params['cv_split']}_epochs{params['epochs']}")

        for idx in tqdm(range(len(dset_val))):
            model.eval()
            image_model_input, mask, inst_seg = dset_val[idx]
            true_img, _, _ = dset_val_notransform[idx]
            model_out = model(image_model_input.unsqueeze(0).to(params['device']))
            model_out = model_out.cpu().detach().numpy().squeeze()
            plot_circles_and_roi_points(true_img, inst_seg, path_analysis_output_cv)
            visualize_image_and_model_output(true_img, model_out, inst_seg, path_analysis_output_cv)


def main():
    parser = argparse.ArgumentParser(description='Train Cell Type Detector')
    parser.add_argument('--df_path', type=str, required=True, help='Path to the dataframe containing the dataset information')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--cv_split', type=int, required=True, help='Cross-validation split index')
    parser.add_argument('--mask_col', type=str, required=True, help='Column name for mask paths')
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--aug_mult', type=float, default=1.0, help='Augmentation multiplier')
    parser.add_argument('--encoder_name', type=str, default='timm-efficientnet-b5', help='Encoder name for the UNet')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes for the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on ("cuda" or "cpu")')
    parser.add_argument('--do_elastic', action='store_true', help='Use elastic transformations')
    parser.add_argument('--do_analyze', action='store_true', help='Perform analysis after training')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable TQDM progress bars')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None, help='Class weights for loss calculation')

    args = parser.parse_args()

    # Convert class_weights from list of floats to appropriate format (e.g., torch.Tensor)
    if args.class_weights is not None:
        args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32)
    else:
        args.class_weights = torch.tensor([0.0001, 1, 1, 1], dtype=torch.float32)

    # Prepare parameters dictionary to pass to the training function
    params = vars(args)
    params['output_dir'] = Path(params['output_dir'])  # Ensure output_dir is a Path object

    # Call the training function
    train_cell_type_detector(params)

if __name__ == '__main__':
    main()
