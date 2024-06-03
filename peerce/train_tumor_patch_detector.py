import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import time
import datetime
import pickle
from peerce.deep_learning.unet_functionality import NInputChanUnet, train_more_valid_unet
from peerce.data_processing.wsi_patch_df_6channel_dataset import WSIPatchDfDataset6channel, get_more_augmented_train_transform_6chan_alb, get_valid_transform_6chan_alb
from peerce.deep_learning.dl_utils import save_model
from peerce.data_processing.datasets import get_data_loaders

import argparse
from peerce.utils.params import train_tumor_patch_detector_params


def train_tumor_patch_detector(train_tumor_patch_detector_params):
    # Extract parameters
    column_path_data = train_tumor_patch_detector_params['column_path_data']
    column_path_mask = train_tumor_patch_detector_params['column_path_mask']
    column_scalar_label = train_tumor_patch_detector_params['column_scalar_label']
    lr = train_tumor_patch_detector_params['lr']
    epochs = train_tumor_patch_detector_params['epochs']
    bs = train_tumor_patch_detector_params['bs']
    aug_mult = train_tumor_patch_detector_params['aug_mult']
    label_smoothing = train_tumor_patch_detector_params['label_smoothing']
    encoder_name = train_tumor_patch_detector_params['encoder_name']
    cv_split = train_tumor_patch_detector_params['cv_split']
    num_workers = train_tumor_patch_detector_params['num_workers']
    do_cosine_annealing = train_tumor_patch_detector_params['do_cosine_annealing']

    df_path = train_tumor_patch_detector_params['data_df_path']
    output_dir = train_tumor_patch_detector_params['output_dir']
    output_dir = Path(output_dir)
    
    date_str = datetime.datetime.now().strftime('%d%b%y').lower()
    output_dir = output_dir / f"train_tumor_patch_detector_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU is not supported.")
        return
    
    # Load the dataframe
    df = pd.read_feather(df_path)

    # Prepare datasets
    dataset_train = WSIPatchDfDataset6channel(patch_df=df, cv_split=cv_split, columns=[column_path_data], use_masks=True, 
                                    transform=get_more_augmented_train_transform_6chan_alb(512, mult=aug_mult), mask_col=column_path_mask,
                                    angio='all', column_scalar_label=column_scalar_label)
    dataset_valid = WSIPatchDfDataset6channel(patch_df=df, cv_split=cv_split, columns=[column_path_data], use_masks=True, 
                                    transform=get_valid_transform_6chan_alb(512), valid=True, mask_col=column_path_mask,
                                    angio='all', column_scalar_label=column_scalar_label)

    # Data loaders
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size=bs, num_workers=num_workers)

    # Model, criterion, and optimizer
    model = NInputChanUnet(n_channels_in=3, model_cls=None, encoder_name=encoder_name, encoder_weights="imagenet", 
                           in_channels=3, classes=2, activation=None).cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if do_cosine_annealing else optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Training loop
    start_time = time.perf_counter()
    loss_acc_dict = None
    for epoch in range(epochs):
        loss_acc_dict = train_more_valid_unet(model, train_loader, valid_loader, optimizer, criterion, loss_acc_dict=loss_acc_dict, disable_tqdm=False)
        scheduler.step()

    date_str = datetime.datetime.now().strftime('%d%b%y').lower()
    info = f"unet_{encoder_name}_cv{cv_split}_{date_str}"
    
    loss_acc_dict['info'] = info
    loss_acc_dict['column_name_data'] = column_path_data
    loss_acc_dict['column_name_mask'] = column_path_mask
    loss_acc_dict['lr'] = lr
    loss_acc_dict['epochs'] = epochs
    loss_acc_dict['bs'] = bs
    loss_acc_dict['aug_mult'] = aug_mult
    loss_acc_dict['label_smoothing'] = label_smoothing
    loss_acc_dict['encoder_name'] = encoder_name
    loss_acc_dict['cv_split'] = cv_split
    loss_acc_dict['num_workers'] = num_workers
    loss_acc_dict['df_path'] = str(df_path)
    
    end_time = time.perf_counter()
    passed_time = str(datetime.timedelta(seconds=round(end_time - start_time)))
    loss_acc_dict['passed_time'] = passed_time
    
    save_model(epochs, model, f'unet_{encoder_name}', optimizer, criterion, output_dir, info)
    # Save plots if necessary
    pickle.dump(loss_acc_dict, open(output_dir / f'loss_acc_dict_{info}.pckl', 'wb'))

    print('TRAINING COMPLETE')
    print(f"Took {passed_time} to train {epochs} epochs with {encoder_name} encoder.")
    

def main():
    parser = argparse.ArgumentParser(description='Train Tumor Patch Detector')

    # Mandatory parameters
    parser.add_argument('--data_df_path', type=str, required=False, help='Path to the dataframe containing the dataset information')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save training outputs')

    # Optional parameters with defaults from train_tumor_patch_detector_params
    parser.add_argument('--column_path_data', type=str, default=train_tumor_patch_detector_params['column_path_data'], help='Column name for data patches paths')
    parser.add_argument('--column_path_mask', type=str, default=train_tumor_patch_detector_params['column_path_mask'], help='Column name for mask patches paths')
    parser.add_argument('--column_scalar_label', type=str, default=train_tumor_patch_detector_params['column_scalar_label'], help='Column name for scalar labels. If "dummy", use dummy targets')
    parser.add_argument('--lr', type=float, default=train_tumor_patch_detector_params['lr'], help='Learning rate')
    parser.add_argument('--epochs', type=int, default=train_tumor_patch_detector_params['epochs'], help='Number of epochs')
    parser.add_argument('--bs', type=int, default=train_tumor_patch_detector_params['bs'], help='Batch size')
    parser.add_argument('--aug_mult', type=float, default=train_tumor_patch_detector_params['aug_mult'], help='Augmentation multiplier')
    parser.add_argument('--label_smoothing', type=float, default=train_tumor_patch_detector_params['label_smoothing'], help='Label smoothing')
    parser.add_argument('--encoder_name', type=str, default=train_tumor_patch_detector_params['encoder_name'], help='Encoder name for the model')
    parser.add_argument('--cv_split', type=int, default=train_tumor_patch_detector_params['cv_split'], help='Cross-validation split index')
    parser.add_argument('--num_workers', type=int, default=train_tumor_patch_detector_params['num_workers'], help='Number of workers for data loading')
    parser.add_argument('--do_cosine_annealing', type=lambda x: (str(x).lower() == 'true'), default=train_tumor_patch_detector_params['do_cosine_annealing'], help='Whether to use cosine annealing')

    args = parser.parse_args()
    
    # Update the parameters dictionary with arguments from the command line
    train_tumor_patch_detector_params.update(vars(args))
    
    # # Temporarily update params for testing
    train_tumor_patch_detector_params['data_df_path'] = '/home/fabian/projects/phd/APEDIA/data/example_patch_df.feather'
    train_tumor_patch_detector_params['output_dir'] = '/home/fabian/projects/phd/APEDIA/data/outputs'
    train_tumor_patch_detector_params['epochs'] = 1
    train_tumor_patch_detector_params['column_path_data'] = 'path_patch_pdl1'
    train_tumor_patch_detector_params['column_path_mask'] = 'path_patch_mask'
    train_tumor_patch_detector_params['column_scalar_label'] = 'tumors'
    
    # Call the training function with updated parameters
    print("Training Tumor Patch Detector")
    train_tumor_patch_detector(train_tumor_patch_detector_params)


if __name__ == '__main__':
    main()
