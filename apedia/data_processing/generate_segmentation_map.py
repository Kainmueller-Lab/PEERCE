from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def visualize_unet_segmentation_array(new_segmentation_array, outpath=None):
    cmap = ListedColormap(['gray', 'white', 'red', 'blue', 'yellow'])

    # Map values from the original range to a new range of positive values
    mapped_array = new_segmentation_array.copy()
    mapped_array[new_segmentation_array == -100] = 0
    mapped_array[new_segmentation_array == 0] = 1
    mapped_array[new_segmentation_array == 1] = 2
    mapped_array[new_segmentation_array == 2] = 3
    mapped_array[new_segmentation_array == 3] = 4

    if outpath is None:
        plt.imshow(mapped_array, cmap=cmap, vmin=0, vmax=4)
        plt.show()
    else:
        plt.imsave(outpath, mapped_array, cmap=cmap)


def create_unet_segmentation_array(path_segmentation_instances_array, save_path=None):
    '''
    Create one_channel segmentation array. Segmentation classes are:
    -100: ignore
    0: background
    1: tz_pos
    2: tz_neg
    3: other
    
    '''
    segmentation_array = np.load(path_segmentation_instances_array)
    # Create a new array with the same shape as segmentation_array & already add background
    new_segmentation_array = (segmentation_array[0] != 0).astype(np.int8)  # background

    # all_cells is to be ignored, if not annotated (-100)
    new_segmentation_array[segmentation_array[0] > 0] = -100  # all_cells

    channel_mapping = {'tz_pos': 1, 'tz_neg': 2, 'other': 3}
    for k, v in channel_mapping.items():
        new_segmentation_array[segmentation_array[v] > 0] = v

    if save_path is not None:
        save_name = f"{''.join(Path(path_segmentation_instances_array).stem.split('_segmentation_channels'))}.npy"
        np.save(save_path / save_name, new_segmentation_array)
        return str(save_path / save_name)
    else:
        visualize_unet_segmentation_array(new_segmentation_array)
        return new_segmentation_array
    
    
def create_segmentation_df(out_path_calculation, n_splits=5, random_state=42):
    """
    Prepare segmentation data for training by collecting file paths, creating cross-validation folds,
    and summarizing in a DataFrame.
    
    Parameters:
    - out_path_calculation: Path to the directory containing segmentation data.
    - n_splits: Number of folds for the KFold cross-validation.
    - random_state: Seed for random operations to ensure reproducibility.
    
    Returns:
    - A pandas DataFrame with paths to the segmentation arrays, png files, wsi names, and cross-validation fold assignments.
    """
    # Find the relevant paths
    paths_segmentation_arrays = sorted(out_path_calculation.glob("*segmentation_channels.npy"))
    paths_segmentation_arrays_multi = sorted(out_path_calculation.glob("*segmentation_channels_multi.npy"))
    paths_png_files = sorted(out_path_calculation.glob("*.png"))
    
    # Convert paths to strings
    string_paths_segmentation_arrays = [str(path) for path in paths_segmentation_arrays]
    string_paths_segmentation_arrays_multi = [str(path) for path in paths_segmentation_arrays_multi]
    string_paths_png_files = [str(path) for path in paths_png_files]

    # Extract unique WSI names
    wsi_names = [Path(s).stem.split('__')[0] for s in string_paths_segmentation_arrays]
    unique_wsi_names = list(set(wsi_names))

    # Create cross-validation folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    element_to_fold = {}
    for fold_number, (_, test_indices) in enumerate(kf.split(unique_wsi_names)):
        for idx in test_indices:
            element_to_fold[unique_wsi_names[idx]] = fold_number

    # Assign each WSI to a fold
    cv_list = [element_to_fold[wsi_name] for wsi_name in wsi_names]
    
    # Summarize all in one DataFrame
    df = pd.DataFrame({
        'path_exact_one_match': string_paths_segmentation_arrays, 
        'path_oneplus_matches': string_paths_segmentation_arrays_multi, 
        'path_patch_png': string_paths_png_files,
        'wsi_name': wsi_names, 'cv_split': cv_list
    })
    
    return df