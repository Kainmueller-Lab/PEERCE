from shutil import copy2
import numpy as np
from scipy.ndimage import binary_fill_holes
import pandas as pd
from pathlib import Path
from skimage import draw
from cellpose import utils as cp_utils
from tqdm import tqdm
import staintools

from peerce.data_processing.segmentation_viz import display_segmentation_channels, plot_circles_and_roi_points



def get_patch_path_all(image_name, df, first_omero_patches, cnn_patch_df):
    # First, try to find the patch in the dataframe 'df'
    df_entry = df[df.pdl1_patch_names == image_name]
    if not df_entry.empty:
        return df_entry['path_patch_pdl1'].iloc[0]

    # If not found, try to find the patch in the 'first_omero_patches' list
    for p in first_omero_patches:
        if p.name == image_name:
            return p

    # If still not found, try to find the patch in the dataframe 'cnn_patch_df'
    cnn_entry = cnn_patch_df[cnn_patch_df.patch_file_out_path.apply(lambda p: Path(p).name) == image_name]
    if not cnn_entry.empty:
        return cnn_entry['patch_file_out_path'].iloc[0]

    # If the patch is still not found after all attempts, print a warning message
    print("Problem: patch not found")


def find_matching_circle_optimized(roi_coords, filled_circles):
    max_count = -1
    best_circle_idx = -1
    count_dict = {}

    # Prepare roi_coords array
    roi_coords = np.array(roi_coords, dtype=float)
    roi_coords = np.round(roi_coords).astype(int)
    
    # Calculate the count of points inside each circle using NumPy broadcasting
    counts = np.sum(filled_circles[:, roi_coords[:, 1], roi_coords[:, 0]], axis=1)

    # Find the best circle index and its count
    best_circle_idx = np.argmax(counts)
    max_count = counts[best_circle_idx]

    # Store non-zero counts in the count_dict
    count_dict = {idx: count for idx, count in enumerate(counts) if count > 0}

    # Print results
    if max_count == 0:
        # print("No points found in any circle.")
        best_circle_idx = -1
    # else:
    #     print(f"Circle {best_circle_idx} contains the most points with {max_count} out of {len(roi_coords)} points.")
    #     if len(count_dict) > 1:
    #         print("Other cells were also matched")

    return best_circle_idx, count_dict


def string_to_float_coords(coord_string):
    return [[float(num) for num in (coords.split(','))] for coords in coord_string.split(' ')]


def get_corresponding_text(roi_df, tuple_roi_coords, image_name=None, replacements=None):
    if image_name is None:
        image_name = roi_df.image_name.unique()[0]
    filtered_roi_df = roi_df[roi_df.image_name == image_name]

    # Create a dictionary with the original (non-unique) roi_coords as keys and the corresponding text as values
    roi_text_dict = {}
    for idx, row in filtered_roi_df.iterrows():
        # sometimes, row['Points'] can be NaN. In that case, skip the row
        if pd.isna(row['Points']):
            continue
        float_coords = string_to_float_coords(row['Points'])
        tuple_coords = tuple(map(tuple, float_coords))
        roi_text_dict[tuple_coords] = row['text']

    # Replace text values based on the corrected lists
    if replacements is None:
        replacements = {
            'TZ neg.': 'tz_neg',
            'TZ neg': 'tz_neg',
            'TZ pos.': 'tz_pos',
            'TZ pos': 'tz_pos',
            'TZ Pos': 'tz_pos',
            'Neutrophiler Granulozyt': 'other',
            'Keine TZ': 'other',
            'Kein TZ': 'other',
            # edit 4apr23: exclude these. Other would declare them as a healthy non-tumor cell
            'Tumorzelle': 'exclude',
            'Eisenpigment': 'exclude',
            # edit 19sep23: exclude these, as well
            'KeineTZ': 'other',
            'Keine TZ pos': 'other'
        }


    roi_text_dict = {key: replacements[val] for key, val in roi_text_dict.items()}

    # Use a list comprehension to find the text for each unique tuple_roi_coords element based on the dictionary
    corresponding_text = [roi_text_dict[coords] for coords in tuple_roi_coords]

    return corresponding_text


def create_results_df(roi_df, outlines_list, roi_coord_list, image_name, replacements=None):
    # Create an array to store filled circles
    filled_circles = np.zeros((len(outlines_list), 512, 512), dtype=bool)

    for idx, coords in enumerate(outlines_list):
        # Create a binary image for the circle and fill it
        filled_circles[idx, coords[:, 1], coords[:, 0]] = 1

    filled_circles = np.array([binary_fill_holes(circle) for circle in filled_circles])

    # Create a unique roi coord list
    # Convert each sublist to a tuple
    tuple_roi_coords = [tuple(map(tuple, sublist)) for sublist in roi_coord_list]

    # Remove duplicates using a set
    tuple_roi_coords = list(set(tuple_roi_coords))

    # Initialize an empty DataFrame with the specified columns
    results_df = pd.DataFrame(columns=['image_name', 'label', 'match_found', 'matched_outlines', 'exactly_one_match', 'idx', 'roi_coords'])

    # image_name = roi_df.image_name.unique()[0]
    corresponding_text = get_corresponding_text(roi_df, tuple_roi_coords, image_name, replacements=replacements)

    # Iterate through each list of test_roi_coords
    for i, (roi_coords, label) in enumerate(zip(tuple_roi_coords, corresponding_text)):
        best_circle_idx, count_dict = find_matching_circle_optimized(roi_coords, filled_circles)
        
        matched_outlines = []
        exactly_one_match = False
        match_found = False

        if best_circle_idx != -1:
            match_found = True
            if len(count_dict) > 1:
                matched_outlines = [outlines_list[idx] for idx in count_dict.keys()]
            else:
                matched_outlines = [outlines_list[best_circle_idx]]
                exactly_one_match = True

        # Add a new row to the DataFrame
        new_row = pd.DataFrame({
            'image_name': [image_name],
            'label': [label],
            'match_found': [match_found],
            'matched_outlines': [matched_outlines],
            'exactly_one_match': [exactly_one_match],
            'idx': [i],
            'roi_coords': [roi_coords]
        })

        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df.convert_dtypes()


def create_segmentation_channels(outlines_list, results_df, include_multiple_matches=False):
    # Initialize an empty array with dimensions (5, 512, 512)
    segmentation_channels = np.zeros((5, 512, 512), dtype=int)

    # Initialize instance number
    instance_number = 1

    # Iterate through outlines_list
    for coords in outlines_list:
        # Create an empty binary image for the current outline
        binary_outline = np.zeros((512, 512), dtype=bool)
        binary_outline[coords[:, 1], coords[:, 0]] = True
        filled_outline = binary_fill_holes(binary_outline)

        # Find the corresponding row in results_df for the current outline
        matched_row = results_df[results_df['matched_outlines'].apply(lambda x: any(np.array_equal(coords, y) for y in x))]
        if not matched_row.empty:
            row = matched_row.iloc[0]
            roi_coords = row['roi_coords']

            # Draw the ROI coordinates in the fifth channel
            for i in range(len(roi_coords) - 1):
                rr, cc = draw.line(int(round(roi_coords[i][1])), int(round(roi_coords[i][0])), int(round(roi_coords[i+1][1])), int(round(roi_coords[i+1][0])))
                segmentation_channels[4, rr, cc] = instance_number
            rr, cc = draw.line(int(round(roi_coords[-1][1])), int(round(roi_coords[-1][0])), int(round(roi_coords[0][1])), int(round(roi_coords[0][0])))
            segmentation_channels[4, rr, cc] = instance_number

            if len(row['matched_outlines']) > 1 and not include_multiple_matches:
                instance_number += 1
                continue

            channel_mapping = {'tz_pos': 1, 'tz_neg': 2, 'other': 3}
            channel_idx = channel_mapping.get(row['label'], None)

            if channel_idx is not None:
                segmentation_channels[channel_idx][filled_outline] = instance_number

        segmentation_channels[0][filled_outline] = instance_number
        instance_number += 1

    return segmentation_channels

 
def process_patch(patch, roi_df, roi_coord_list, image_name, model, replacements=None):
    masks, flows, styles, diams = model.eval(patch, diameter=15, channels=[0,3], invert=True, augment=True, net_avg=True,
                                            flow_threshold=0.4)

    outlines_list = cp_utils.outlines_list(masks)
    # outlines = utils.masks_to_outlines(masks)

    results_df = create_results_df(roi_df, outlines_list, roi_coord_list, image_name, replacements=replacements)
    return results_df, outlines_list

def create_cellpose_instance_segmentations_add_rois(path_folder_patch_imgs, out_path_calculation, out_path_viz, roi_df, model, make_hema, viz, tip_the_balance, replacements=None):
    for idx, image_name in enumerate(tqdm(roi_df['image_name'].unique())):
        # if idx < 18:
        #     continue
        # # only to 20 patches for testing
        # if idx > 20:
        #     break
        path_patch = path_folder_patch_imgs / image_name
        if not path_patch.is_file():
            print(f"Patch {path_patch} does not exist.")
            continue
        
        copy2(path_patch, out_path_calculation)
        patch = staintools.read_image(str(path_patch))
        roi_coord_list = [string_to_float_coords(i) for i in roi_df[roi_df.image_name == image_name].Points if not pd.isna(i)]
        
        results_df_patch, outlines_list_patch = process_patch(patch, roi_df, roi_coord_list, image_name, model, replacements=replacements)
        results_df_hema, outlines_list_hema = process_patch(make_hema(patch), roi_df, roi_coord_list, image_name, model, replacements=replacements)
        
        if results_df_patch['match_found'].sum() >= results_df_hema['match_found'].sum() + tip_the_balance:
            results_df, outlines_list = results_df_patch, outlines_list_patch
        else:
            results_df, outlines_list = results_df_hema, outlines_list_hema

        segmentation_channels = create_segmentation_channels(outlines_list, results_df, include_multiple_matches=False)
        segmentation_channels_multi = create_segmentation_channels(outlines_list, results_df, include_multiple_matches=True)
        
        np.save(out_path_calculation / f"{Path(image_name).stem}_segmentation_channels.npy", segmentation_channels)
        np.save(out_path_calculation / f"{Path(image_name).stem}_segmentation_channels_multi.npy", segmentation_channels_multi)
        results_df.to_pickle(out_path_calculation / f"{Path(image_name).stem}_results_df.pckl")
        
        if viz:
            display_segmentation_channels(segmentation_channels, image_name, save_path=out_path_viz)
            plot_circles_and_roi_points(outlines_list, results_df, patch, save_path=out_path_viz / f"{Path(image_name).stem}.png")