from cellpose import models
import pandas as pd
from pathlib import Path
import staintools
from tqdm import tqdm
import numpy as np
from shutil import copy2
import re

from apedia.data_processing.roi_infos import print_roi_infos
from apedia.data_processing.deepliif_hema_patch import MakeHemaPatch
from apedia.data_processing.instance_segmentation import process_patch, create_segmentation_channels
from apedia.data_processing.instance_segmentation import string_to_float_coords, get_patch_path_all
from apedia.data_processing.segmentation_viz import display_segmentation_channels, plot_circles_and_roi_points

def preprocess_cell_type_data(out_path, path_df, path_cnn_pred_patches_df, path_roi_csv, first_omero_patches,
                              path_roi_csv_cnn=None, roi_infos=False, tip_the_balance=0):
    """
    Preprocess cell type data using Cellpose models.

    Parameters:
    - out_path: Directory where output files will be saved.
    - path_df: Path to the main DataFrame containing patch information.
    - path_cnn_pred_patches_df: Path to the DataFrame containing CNN prediction patches.
    - path_roi_csv: Path to the CSV file containing ROI information.
    - path_roi_csv_cnn: Optional. Path to the CSV file containing additional ROI information from CNN predictions.
    """
    
    # Load data frames
    df = pd.read_feather(path_df)
    cnn_patch_df = pd.read_feather(path_cnn_pred_patches_df)
    roi_df = pd.read_csv(path_roi_csv)
    if path_roi_csv_cnn:
        cnn_roi_df = pd.read_csv(path_roi_csv_cnn)
        # Combine ROI DataFrames if necessary
        roi_df = pd.concat([roi_df, cnn_roi_df], ignore_index=True)
    
    
    if roi_infos:
        print_roi_infos(roi_df)
    # Prepare output directories
    out_path = Path(out_path)
    out_path_viz = out_path / 'viz'
    out_path_calculation = out_path / 'calculation'
    out_path_viz.mkdir(parents=True, exist_ok=True)
    out_path_calculation.mkdir(parents=True, exist_ok=True)
    
    # Load and configure the Cellpose model
    model = models.Cellpose(gpu=True, model_type='nuclei')
    
    # also load make_hema
    make_hema = MakeHemaPatch()
    # Process each image in the ROI DataFrame
    for idx, image_name in enumerate(tqdm(roi_df['image_name'].unique())):
        # Get the patch path and load the patch
        path_patch = get_patch_path_all(image_name, df, first_omero_patches, cnn_patch_df)
        copy2(path_patch, out_path_calculation)
        patch = staintools.read_image(str(path_patch))
        roi_coord_list = [string_to_float_coords(i) for i in roi_df[roi_df.image_name == image_name].Points if not pd.isna(i)]
        
        results_df_patch, outlines_list_patch = process_patch(patch, roi_df, roi_coord_list, image_name, model)
        results_df_hema, outlines_list_hema = process_patch(make_hema(patch), roi_df, roi_coord_list, image_name, model)
        
        # Comparison logic
        if results_df_patch['match_found'].sum() >= results_df_hema['match_found'].sum() + tip_the_balance:
            results_df, outlines_list = results_df_patch, outlines_list_patch
            print("Best results are from the original patch", f"Out of {len(results_df)} ROIs, {results_df['match_found'].sum()}",
                  f"ROIs found at least one cell nucleus (vs {results_df_hema['match_found'].sum()} ROIs).")
            print("Best results are from the original patch", f"Out of {len(results_df)} ROIs, {results_df['exactly_one_match'].sum()}",
                  f"ROIs found exactly one cell nucleus (vs {results_df_hema['exactly_one_match'].sum()} ROIs).")
        else:
            results_df, outlines_list = results_df_hema, outlines_list_hema
            print("Best results are from the Hema patch", f"Out of {len(results_df)} ROIs, {results_df['match_found'].sum()} ROIs found at least one cell nucleus (vs {results_df_patch['match_found'].sum()} ROIs).")
            print("Best results are from the Hema patch.", f"Out of {len(results_df)} ROIs, {results_df['exactly_one_match'].sum()} ROIs found exactly one cell nucleus (vs {results_df_patch['exactly_one_match'].sum()} ROIs).")


        segmentation_channels = create_segmentation_channels(outlines_list, results_df, include_multiple_matches=False)
        segmentation_channels_multi = create_segmentation_channels(outlines_list, results_df, include_multiple_matches=True)
        
        # Save results
        np.save(out_path_calculation / f"{Path(image_name).stem}_segmentation_channels.npy", segmentation_channels)
        np.save(out_path_calculation / f"{Path(image_name).stem}_segmentation_channels_multi.npy", segmentation_channels_multi)
        results_df.to_pickle(out_path_calculation / f"{Path(image_name).stem}_results_df.pckl")

        display_segmentation_channels(segmentation_channels, image_name, save_path=out_path_viz)
        plot_circles_and_roi_points(outlines_list, results_df, patch, save_path=out_path_viz / f"{Path(image_name).stem}.png")

# Example call to the function

path_other_patches = Path("/home/fabian/projects/phd/ssd_data/first_patches_omero_upload")
first_omero_patches = list(path_other_patches.rglob("*.png"))
first_omero_patches = [p for p in first_omero_patches if re.match(r".*W\d+.*PD-L1.*.png", p.name)]

hema_tip_the_balance = np.Inf
original_tip_the_balance = -np.Inf
fair_balance = 0

tip_the_balance = fair_balance


preprocess_cell_type_data(
    out_path="/path/to/output",
    path_df="/path/to/main_dataframe.feather",
    path_cnn_pred_patches_df="/path/to/cnn_predictions.feather",
    path_roi_csv="/path/to/roi_info.csv",
    path_roi_csv_cnn="/path/to/additional_roi_info.csv",
    first_omero_patches=first_omero_patches,
    tip_the_balance=tip_the_balance,
)
