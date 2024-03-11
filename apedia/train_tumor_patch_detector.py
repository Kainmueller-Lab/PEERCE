import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from apedia.deep_learning.unet_functionality import NInputChanUnet, train_more_valid_unet
from apedia.data_processing.wsi_patch_df_6channel_dataset import WSIPatchDfDataset6channel, get_more_augmented_train_transform_6chan_alb, get_valid_transform_6chan_alb
from apedia.deep_learning.dl_utils import save_model
from apedia.data_processing.datasets import get_data_loaders
import time
import datetime
import pickle


def train_tumor_patch_detector(train_tumor_patch_detector_params):
    # Extract parameters
    column_name_data = train_tumor_patch_detector_params['column_name_data']
    column_name_mask = train_tumor_patch_detector_params['column_name_mask']
    lr = train_tumor_patch_detector_params['lr']
    epochs = train_tumor_patch_detector_params['epochs']
    bs = train_tumor_patch_detector_params['bs']
    aug_mult = train_tumor_patch_detector_params['aug_mult']
    label_smoothing = train_tumor_patch_detector_params['label_smoothing']
    encoder_name = train_tumor_patch_detector_params['encoder_name']
    cv_split = train_tumor_patch_detector_params['cv_split']
    num_workers = train_tumor_patch_detector_params['num_workers']
    do_cosine_annealing = train_tumor_patch_detector_params['do_cosine_annealing']

    df_path = train_tumor_patch_detector_params['path_dataframe']
    output_dir = train_tumor_patch_detector_params['output_dir']
    output_dir = Path(output_dir)

    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU is not supported.")
        return
    
    # Load the dataframe
    df = pd.read_feather(df_path)

    # Prepare datasets
    dataset_train = WSIPatchDfDataset6channel(patch_df=df, cv_split=cv_split, columns=[column_name_data], use_masks=True, 
                                    transform=get_more_augmented_train_transform_6chan_alb(512, mult=aug_mult), mask_col=column_name_mask,
                                    angio='all')
    dataset_valid = WSIPatchDfDataset6channel(patch_df=df, cv_split=cv_split, columns=[column_name_data], use_masks=True, 
                                    transform=get_valid_transform_6chan_alb(512), valid=True, mask_col=column_name_mask,
                                    angio='all')

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

    date_str = datetime.now().strftime('%d%b%y').lower()
    info = f"unet_{encoder_name}_cv{cv_split}_{date_str}"
    
    loss_acc_dict['info'] = info
    loss_acc_dict['column_name_data'] = column_name_data
    loss_acc_dict['column_name_mask'] = column_name_mask
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
    
    output_dir.mkdir(exist_ok=True)
    save_model(epochs, model, f'unet_{encoder_name}', optimizer, criterion, output_dir, info)
    # Save plots if necessary
    pickle.dump(loss_acc_dict, open(output_dir / f'loss_acc_dict_{info}.pckl', 'wb'))

    print('TRAINING COMPLETE')
    print(f"Took {passed_time} to train {epochs} epochs with {encoder_name} encoder.")