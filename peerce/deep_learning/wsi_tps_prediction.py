from pathlib import Path
import fnmatch
import pandas as pd
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import scipy.ndimage
from scipy.ndimage import binary_fill_holes
from scipy.special import softmax

import gc

# cellpose related functionality
from skimage import io
from cellpose import utils


from peerce.deep_learning.unet_functionality import NInputChanUnet
from peerce.data_processing.wsi_patch_df_6channel_dataset import get_valid_transform_6chan_alb


def get_matching_model_path(df, model_paths, wsi_name):
    # filter the DataFrame based on the condition
    mask = df['path_pdl1s'].apply(lambda x: Path(x).name) == wsi_name
    filtered_df = df[mask]

    # get the cv split for the filtered row
    if filtered_df.empty:
        # use cv_split of 0 as default
        cv_split = 0
    else:
        cv_split = filtered_df['cv_split'].iloc[0]

    # find the model patch that matches the cv split
    matching_model_path = fnmatch.filter([str(path) for path in model_paths], f"*cv_split{cv_split}_*")[0]

    return Path(matching_model_path)


def get_unet_wsi_dataset_prediction(wsi_dataset, model, idx):
    img, loc = wsi_dataset[idx]
    segmentation = model.predict_segmentation(img)
    prediction = segmentation.mean()
    return prediction, loc, segmentation

def identify_tumor_patches_wsi(wsi_dataset, model, wsi_name_id, wsi_prediction_folder, threshold = 0.6, viz=True):
    prediction_folder_tumor = wsi_prediction_folder / 'tumor_patches'
    prediction_folder_tumor.mkdir(parents=True, exist_ok=True)
    if viz:
        prediction_folder_tumor_viz = wsi_prediction_folder / 'tumor_patches_segmentation_viz'
        prediction_folder_tumor_viz.mkdir(parents=True, exist_ok=True)

    preds, locs, coverages, mask_coverages, idxs, path_patch = [], [], [], [], [], []
    for i in tqdm(range(len(wsi_dataset))):
        pred, y, segmentation = get_unet_wsi_dataset_prediction(wsi_dataset, model, i)
        # mask_coverage = m.float().mean().item()
        mask_coverage = -1
        patch, location = wsi_dataset.get_patch(i)
        if pred > threshold:
            imageio.imwrite(prediction_folder_tumor / f"{wsi_name_id}_{int(location[0])}_{int(location[1])}.png", patch)
            if viz:
                plot_tumor_segmentation_map(patch, segmentation, wsi_name_id, location,
                                            outpath=prediction_folder_tumor_viz / f"{wsi_name_id}_{int(location[0])}_{int(location[1])}_segmentation_viz.jpg")
            path_patch.append(prediction_folder_tumor / f"{wsi_name_id}_{int(location[0])}_{int(location[1])}.png")
        else:
            path_patch.append(None)
        preds.append(pred)
        locs.append(y)
        coverages.append(-1)
        idxs.append(i)
        mask_coverages.append(mask_coverage)
        # print(pred, y)

    df = pd.DataFrame({'idx': idxs, 'prediction': preds, 'location': locs, 
                        # 'tumor_coverage': [c[0] for c in coverages], 
                        # 'non_tumor_coverage': [c[1] for c in coverages], 
                        # 'tissue_coverage': [c[1] for c in coverages], 
                        'mask_coverage': mask_coverages,
                        'path_patch': path_patch})
    df['path_patch'] = df['path_patch'].astype(str)
    df.to_feather(wsi_prediction_folder / f"{wsi_name_id}_tumor_patch_prediction_df.feather")
    return df



def get_cellpose_outlines(patch, cellpose_model, make_hema):
        
    masks, flows, styles, diams = cellpose_model.eval(patch, diameter=15, channels=[0,3], invert=True, augment=True, net_avg=True,
                                            flow_threshold=0.4)
    outlines_list = utils.outlines_list(masks)

    masks, flows, styles, diams = cellpose_model.eval(make_hema(patch), diameter=15, channels=[0,3], invert=True, augment=True, net_avg=True,
                                            flow_threshold=0.4)
    outlines_list_hema = utils.outlines_list(masks)
    
    # use outline list with most detected outlines
    if len(outlines_list_hema) > len(outlines_list):
        return outlines_list_hema
    else:
        return outlines_list
    
    
def get_cellpose_instance_masks(patch, cellpose_model, make_hema):
    '''
    Offers some speedup compared to "get_cellpose_outlines" by not computing the outlines.
    '''
    masks, flows, styles, diams = cellpose_model.eval(patch, diameter=15, channels=[0,3], invert=True, augment=True, net_avg=True,
                                            flow_threshold=0.4)
    # outlines_list = utils.outlines_list(masks)

    masks_hema, flows, styles, diams = cellpose_model.eval(make_hema(patch), diameter=15, channels=[0,3], invert=True, augment=True, net_avg=True,
                                            flow_threshold=0.4)
    # outlines_list_hema = utils.outlines_list(masks)
    
    # use outline list with most detected outlines
    if len(np.unique(masks_hema)) > len(np.unique(masks)):
    # if len(outlines_list_hema) > len(outlines_list):
        # print("Use hema")
        return masks_hema
    else:
        return masks
    
def get_binary_mask_from_outline(coords):
    binary_outline = np.zeros((512, 512), dtype=bool)
    binary_outline[coords[:, 1], coords[:, 0]] = True
    filled_outline = binary_fill_holes(binary_outline)
    return filled_outline


def create_cell_outline_arrays_from_instance_masks(instance_masks):
    '''
    Offers a significant speedup, as we now do not need to fill all the outlines, which took ~4 seconds per patch
    '''
    outline_arr_list = []
    for inst_num in np.unique(instance_masks):
        # 0 is all cells, so skip
        if inst_num == 0:
            continue
        outline_arr_list.append(instance_masks == inst_num)
    outline_arrs = np.stack(outline_arr_list)
    return outline_arrs



def get_correct_model_segmentation(tumor_patch_df, paths_segmentation_models, segmentation_df):
    # assuming that all patches in tumor_patch_df are from the same WSI
    path_patch = tumor_patch_df.path_patch.iloc[0]
    patch_wsi_name = re.split(r'_\d+_\d+', Path(path_patch).stem)[0]

    cv_split_df_column = segmentation_df[segmentation_df['wsi_name'] == patch_wsi_name].cv
    if len(cv_split_df_column) == 0:
        # use cv_split of 0 as default, if the WSI is not in the segmentation_df
        cv_split = 0
    else:
        cv_split = cv_split_df_column.iloc[0]

    correct_model_path = fnmatch.filter([str(sm) for sm in paths_segmentation_models], f"*cv_split{cv_split}*")[0]
    checkpoint = torch.load(correct_model_path, map_location='cuda')
    model = NInputChanUnet(n_channels_in=3, model_cls=None, encoder_name="timm-efficientnet-b5", encoder_weights="imagenet", 
                        in_channels=3, classes=4, activation=None)
    model = model.to('cuda')
    model.eval()
    load_status = model.load_state_dict(checkpoint['model_state_dict'])
    print(load_status)
    return model



class TpsSegmentationDataset(Dataset):
    """Create dataset from tumor patches dataframe."""

    def __init__(self, df, transform=get_valid_transform_6chan_alb(512)):
        self.df = df
        self.patches = self.df.path_patch.values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patch = io.imread(self.patches[idx])
        

        if self.transform:
            transformed = self.transform(image=patch)
            patch = transformed['image']
            patch = patch.cuda()

        return patch
    
    def get_img(self, idx):
        patch = io.imread(self.patches[idx])
        return patch
    
    def get_patch_path(self, idx):
        return self.patches[idx]
    
    


def create_cell_outline_arrays(outline_list):
    outline_arr_list = []
    for outline in outline_list:
        outline_arr_list.append(get_binary_mask_from_outline(outline))
    outline_arrs = np.stack(outline_arr_list)
    return outline_arrs


def get_cell_type_segmentation_map(model, patch_tensor):
    model_prediction = model(patch_tensor.unsqueeze(0).cuda())
    model_prediction = model_prediction.squeeze().cpu().detach().numpy()
    model_prediction = softmax(model_prediction, axis=0)
    return model_prediction


def get_cell_type_predictions(outline_arrs, prediction):
    # Multiply outline arrays with the softmax predictions
    masked_prediction = outline_arrs[:, None, :, :] * prediction[None, :, :, :]
    
    # Sum over the spatial dimensions for each channel
    summed_values = np.sum(masked_prediction, axis=(-1, -2))
    
    # Normalize to make the sum equal to 1 along the channel axis
    norm_values = summed_values / np.sum(summed_values, axis=-1, keepdims=True)
    # cut out background now. Can be reconstructed by 1 - sum of all other classes
    norm_values = norm_values[:, 1:]
    
    # Get the cell type predictions based on the maximum value along the channel axis
    cell_type_predictions = np.argmax(norm_values, axis=-1)
    
    return cell_type_predictions, norm_values


def create_detailed_df(filtered_df, mean_cell_prediction_softmax_list):
    # Create empty lists to store the new rows
    new_rows = []
    
    # Iterate through each row in the original DataFrame and the corresponding softmax predictions
    for idx, softmax_array in zip(filtered_df.index, mean_cell_prediction_softmax_list):
        path_patch_value = filtered_df.loc[idx, 'path_patch']
        
        # Create new rows for the detailed DataFrame
        new_rows.extend([[path_patch_value, *softmax_value, np.max(softmax_value), np.argmax(softmax_value)] for softmax_value in softmax_array])
    
    # Create the new detailed DataFrame
    filtered_df_detailed = pd.DataFrame(new_rows, columns=['path_patch', 'TC+', 'TC-', 'Other Cell', 'Max Softmax', "class_prediction_0pos_1neg_2other"])
    
    return filtered_df_detailed


# combined into one function:
def predict_cell_types_wsi(dset_tps, cellpose_model, make_hema, model_seg, filtered_df, 
                           wsi_prediction_folder, wsi_name_id):
    cell_type_predictions_list = []
    mean_cell_prediction_softmax_list = []
    for i in tqdm(range(len(dset_tps))):
        patch_tensor = dset_tps[i]
        patch_img = dset_tps.get_img(i)
        path_patch_img = dset_tps.get_patch_path(i)
        # also works, but much slower:
        # outline_list = get_cellpose_outlines(patch_img, cellpose_model, make_hema)
        # outline_arrs = create_cell_outline_arrays(outline_list)
        # Major speedup:
        instance_masks = get_cellpose_instance_masks(patch_img, cellpose_model, make_hema)
        instance_mask_arrs = create_cell_outline_arrays_from_instance_masks(instance_masks)
        cell_type_segmentation_map = get_cell_type_segmentation_map(model_seg, patch_tensor)
        cell_type_predictions, mean_cell_prediction_softmax = get_cell_type_predictions(instance_mask_arrs, cell_type_segmentation_map)
        cell_type_predictions_list.append(cell_type_predictions)
        mean_cell_prediction_softmax_list.append(mean_cell_prediction_softmax)
        plot_instance_masks(patch_img, instance_mask_arrs, cell_type_predictions, path_patch_img, outpath=wsi_prediction_folder)

    cell_type_predictions_arr = np.concatenate(cell_type_predictions_list)
    # print(np.unique(cell_type_predictions_arr, return_counts=True))

    cell_type_count_list = [np.unique(wsi_cell_type_preds, return_counts=True)[1] for wsi_cell_type_preds in cell_type_predictions_list]
    cell_type_unique_list = [np.unique(wsi_cell_type_preds, return_counts=True)[0] for wsi_cell_type_preds in cell_type_predictions_list]


    filtered_df['TC+'] = 0
    filtered_df['TC-'] = 0
    filtered_df['Other Cell'] = 0
    for i ,(ct_unique, ct_count) in enumerate(zip(cell_type_unique_list, cell_type_count_list)):
        # print(i, ct_unique, ct_count)
        for unique, count in zip(ct_unique, ct_count):
            if unique == 0:
                filtered_df.loc[i, 'TC+'] = count
            elif unique == 1:
                filtered_df.loc[i, 'TC-'] = count
            elif unique == 2:
                filtered_df.loc[i, 'Other Cell'] = count
            else:
                raise ValueError
            
    filtered_df['TPS'] = filtered_df['TC+'].sum() / (filtered_df['TC+'].sum() + filtered_df['TC-'].sum())
    print(f"TPS is at {filtered_df['TPS'][0]:.2%}")

    # Save filtered_df to Feather
    filtered_df.to_feather(wsi_prediction_folder / f"{wsi_name_id}_cell_type_prediction_df.feather")

    # Create and save filtered_df_detailed
    filtered_df_detailed = create_detailed_df(filtered_df, mean_cell_prediction_softmax_list)
    filtered_df_detailed.to_feather(wsi_prediction_folder / f"{wsi_name_id}_cell_type_prediction_detailed_df.feather")

    # Output path for verification
    print(f"Saved the cell type prediction df to {wsi_prediction_folder / f'{wsi_name_id}_cell_type_prediction_df.feather'}")
    print(f"Saved the detailed cell type prediction df to {wsi_prediction_folder / f'{wsi_name_id}_cell_type_prediction_detailed_df.feather'}")
    
    
    # This is done to address a potential memory leak
    # Free variables
    del cell_type_predictions_list
    del cell_type_predictions_arr
    del cell_type_count_list
    del cell_type_unique_list

    # Clear DataFrame | use dummy instead
    filtered_df = pd.DataFrame()
    filtered_df_detailed = pd.DataFrame()
    
    # Trigger garbage collection
    gc.collect()

    
    return filtered_df


###############################################
# Visualization functions
###############################################

def plot_instance_masks(patch_img, instance_masks, cell_type_predictions, path_patch_img, outpath=None):
    # Define the color map
    cmap = plt.cm.colors.ListedColormap(['black', 'gray', 'red', 'blue', 'yellow'])

    # Prepare cell type mask
    color_mask = np.ones(instance_masks.shape[1:], dtype=np.int8)
    outline_mask = np.zeros(instance_masks.shape[1:], dtype=np.int8)

    # Prepare cell type counters
    cell_counts = [0, 0, 0]  # TC+, TC-, Other

    for idx, cell_type in enumerate(cell_type_predictions):
        mask = instance_masks[idx]

        # Create outline of the cell mask
        dilated_mask = scipy.ndimage.binary_dilation(mask)
        outline = dilated_mask ^ mask
        outline_mask[outline] = 1  # mark the outline

        # Count cell types
        cell_counts[cell_type] += 1

        # Color the cells
        color_mask[mask > 0] = cell_type + 2  # Adding 2 to keep 0 as black (for outlines) and 1 as gray (for background)

    # Overwrite cell colors with black for outline pixels
    color_mask[outline_mask == 1] = 0

    # Create a subplot with two images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the patch image on the left
    axes[0].imshow(patch_img, cmap='gray')
    axes[0].set_title('Patch Image')

    # Display the instance masks on the right with no interpolation
    axes[1].imshow(color_mask, cmap=cmap, vmin=0, vmax=4, interpolation='none')
    axes[1].set_title('Cell Segmentations')

    # Remove axes for cleaner look
    axes[0].axis('off')
    axes[1].axis('off')

    # Add title to the figure
    img_name = Path(path_patch_img).stem
    fig.suptitle(img_name, fontsize=16)

    # Add cell type counts to the bottom
    fig.text(0.5, 0.02, f"Cell types detected: TC+ (red): {cell_counts[0]}, TC- (blue): {cell_counts[1]}, Other Cells (yellow): {cell_counts[2]}", ha='center')

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.06)
    # Save or show the visualization
    if outpath is not None:
        save_dir = outpath / 'cell_type_prediction_viz'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(path_patch_img).stem}_cell_type_prediction.jpg"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
        
def plot_tumor_segmentation_map(patch, segmentation, wsi_name_id, location, outpath=None):
    # Create a grid layout to organize our plot
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    # Display the original image
    ax0.imshow(patch)
    ax0.set_title('Original Image')
    ax0.axis('off')

    # Display the segmentation map
    im = ax1.imshow(segmentation, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Tumor Segmentation')
    ax1.axis('off')

    # Add colorbar
    fig.colorbar(im, cax=ax2)

    # Add title to the figure
    fig.suptitle(f"{wsi_name_id}_{int(location[0])}_{int(location[1])}", fontsize=16)

    # Add tumor coverage score to the bottom, increase fontsize to 14
    fig.text(0.5, 0.02, f"Tumor coverage score is {segmentation.mean():.2%}", ha='center', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.07)

    # Save or show the visualization
    if outpath is not None:
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
