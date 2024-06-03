import argparse
import datetime
import pickle
from pathlib import Path

import pandas as pd
from cellpose import models

from peerce.data_processing.deepliif_hema_patch import MakeHemaPatch
from peerce.data_processing.generate_segmentation_map import create_segmentation_df, create_unet_segmentation_array
from peerce.data_processing.instance_segmentation import create_cellpose_instance_segmentations_add_rois
from peerce.data_processing.roi_infos import print_roi_infos

from peerce.utils.params import preprocess_cell_type_data_params


def preprocess_cell_type_data(output_dir, path_folder_patch_imgs, path_roi_csv,
                              path_roi_csv_2=None, no_roi_infos=False, tip_the_balance=0, replacements=None,
                              path_deepliif_hema_gan_weights=None, viz=False):
    """
    Preprocess cell type data using Cellpose models.

    Parameters:
    - out_path: Directory where output files will be saved.
    - path_folder_patch_imgs: Path to the folder containing patch images.
    - path_roi_csv: Path to the CSV file containing ROI information.
    - path_roi_csv_2: Optional. Path to an additional CSV file containing ROI information.
    - no_roi_infos: Whether to print ROI information. Default is False.
    - tip_the_balance: Balance tipping value for comparison logic between original and Hema patches. Default is 0.
                        If the number of matches in the original patch is greater than the number of matches in the Hema patch plus this value,
                        the original patch is used.
    - replacements: Path to a pickle file containing the replacements dictionary or the dictionary itself. Default is None.
    - path_deepliif_hema_gan_weights: Path to the deepliif hema GAN weights. Default is None.
    - viz: Enable visualization of segmentation channels and ROI points. Default is False.
    """
    # Convert paths to Path objects
    output_dir = Path(output_dir)
    path_folder_patch_imgs = Path(path_folder_patch_imgs)
    path_roi_csv = Path(path_roi_csv)
    if path_roi_csv_2:
        path_roi_csv_2 = Path(path_roi_csv_2)
    if path_deepliif_hema_gan_weights:
        path_deepliif_hema_gan_weights = Path(path_deepliif_hema_gan_weights)

    # Load data frames
    roi_df = pd.read_csv(path_roi_csv)
    if path_roi_csv_2:
        roi_df_2 = pd.read_csv(path_roi_csv_2)
        # Combine ROI DataFrames if necessary
        roi_df = pd.concat([roi_df, roi_df_2], ignore_index=True)
    
    # remove nan text ROIs
    roi_df = roi_df[~roi_df.text.isna()].copy()
    if replacements is not None:
        # Update the dataframe using the replacements dictionary
        roi_df['cell_texts'] = roi_df.text.replace(replacements)
    
    if not no_roi_infos:
        print_roi_infos(roi_df)

    # Prepare output directories
    # add a date string to the output path
    current_date_string = datetime.datetime.now().strftime('%d%b%y').lower()
    output_dir = str(output_dir) / f"cell_type_preprocessing_{current_date_string}"
    output_dir = Path(output_dir)
    out_path_calculation = output_dir / 'calculation'
    out_path_calculation.mkdir(parents=True, exist_ok=True)
    
    out_path_viz = output_dir / 'viz'
    if viz:
        out_path_viz.mkdir(parents=True, exist_ok=True)

    # Load and configure the Cellpose model
    model = models.Cellpose(gpu=True, model_type='nuclei')
    # Load MakeHemaPatch
    make_hema = MakeHemaPatch(path_network_weights=path_deepliif_hema_gan_weights)
    # Process images
    create_cellpose_instance_segmentations_add_rois(path_folder_patch_imgs, out_path_calculation, out_path_viz, roi_df, model, make_hema, viz, tip_the_balance,
                                                    replacements=replacements)

    # Next, create the segmentation df used to train cell segmentation, as well as the segmentation arrays
    df = create_segmentation_df(out_path_calculation)

    
    # create the segmentation arrays
    seg_arrays = []
    seg_arrays_multi = []
    print("Creating segmentation arrays")
    for idx, row in df.iterrows():
        seg_array = create_unet_segmentation_array(row['path_exact_one_match'], out_path_calculation)
        seg_arrays.append(seg_array)
        seg_array_multi = create_unet_segmentation_array(row['path_oneplus_matches'], out_path_calculation)
        seg_arrays_multi.append(seg_array_multi)
        
    # add the paths of the segmentation arrays to the dataframe
    df = df.assign(path_seg_one_match=seg_arrays, path_seg_array_oneplus_matches=seg_arrays_multi)
    
    # save df
    df_out_path = output_dir / f'{current_date_string}_hema_patch_if_more_matches_updated_roi_df.feather'
    print(f"Saving DataFrame to {df_out_path}")
    df.to_feather(df_out_path)


def main():
    parser = argparse.ArgumentParser(description='Preprocess cell type data using Cellpose models and additional processing steps.')

    # Mandatory parameters
    parser.add_argument('--path_roi_csv', type=str, help='Path to the CSV file containing ROI information.', required=False)
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs.', required=False)
    parser.add_argument('--path_folder_patch_imgs', type=str, help='Path to the folder containing patch images.', required=False)


    # Optional parameters
    parser.add_argument('--path_roi_csv_2', type=str, help='Path to the additional CSV file containing ROI information, if any.', default=None)
    parser.add_argument('--no_roi_infos', action='store_true', help='Whether to print ROI information.', default=preprocess_cell_type_data_params["no_roi_infos"])
    parser.add_argument('--tip_the_balance', type=float, help="""Balance tipping value for comparison logic between original and Hema patches. Default is 0.
                        If the number of matches in the original patch is greater than the number of matches in the Hema patch plus this value,
                        the original patch is used.""", default=preprocess_cell_type_data_params["tip_the_balance"])
    parser.add_argument('--replacements', type=str, help='Path to a pickle file containing the replacements dictionary or the dictionary itself.', default=preprocess_cell_type_data_params["replacements"])
    parser.add_argument('--path_deepliif_hema_gan_weights', type=str, help='Path to the deepliif hema GAN weights.', default=preprocess_cell_type_data_params["path_deepliif_hema_gan_weights"])
    parser.add_argument('--viz', action='store_true', help='Enable visualization of segmentation channels and ROI points.', default=preprocess_cell_type_data_params["viz"])

    args = parser.parse_args()
    
    params = vars(args)


    # Check if params['replacements'] is a string (indicating a file path)
    if isinstance(params['replacements'], str):
        with open(params['replacements'], 'rb') as file:
            params['replacements'] = pickle.load(file)

    # Set some parameters manually for now
    params['path_roi_csv'] = Path("/home/fabian/projects/phd/APEDIA/data/example_roi_table.csv")
    params['path_roi_csv'] = "/home/fabian/projects/phd/APEDIA/data/example_roi_table_reduced.csv"
    params['output_dir'] = '/home/fabian/projects/phd/APEDIA/data/outputs/cell_type_preprocessing'
    params['path_folder_patch_imgs'] = Path("/home/fabian/projects/phd/APEDIA/data/example_seg_patches/")

    
    # Call the preprocessing function
    preprocess_cell_type_data(**params)

if __name__ == '__main__':
    main()

