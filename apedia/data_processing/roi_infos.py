
import numpy as np

def print_roi_infos(roi_df):
    wsi_names = roi_df.apply(lambda row: row['image_name'].split('__')[0], axis=1)
    print("Total ROI annotations:")
    unique_names = list(np.unique(wsi_names))
    unique_names = [name.split('_patch')[0] for name in unique_names]
    unique_names = [name for name in unique_names if not name.endswith("_patch")]
    total_unique_wsi = len(set(unique_names))
    print(f"{total_unique_wsi} total WSIs are annotated")
    print(f"{roi_df.image_name.unique().shape[0]} Patches are annotated")
    print(f"{roi_df.roi_id.unique().shape[0]} cells are annotated")