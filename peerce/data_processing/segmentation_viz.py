import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def display_segmentation_channels(segmentation_channels, image_name, save_path=None):
    # Specify the titles for each channel
    titles = [
        'All Outlines',
        'TZ Positive',
        'TZ Negative',
        'Other Cells',
        'ROI Coords'
    ]

    # Calculate the number of rows needed for the subplots
    num_channels = len(segmentation_channels)
    num_rows = (num_channels + 2) // 3

    # Create the subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(9, num_rows * 3))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate through the segmentation channels and display them
    for i, (channel, title) in enumerate(zip(segmentation_channels, titles)):
        axes[i].imshow(channel, cmap='jet')
        axes[i].set_title(title)
        axes[i].axis('off')

    # Hide the extra axes (if any)
    for i in range(len(titles), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Segmentation maps for {Path(image_name).stem}")

    # Save or display the subplots
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path / f"{Path(image_name).stem}_segmentation.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def plot_circles_and_roi_points(circle_outline_list, results_df, image=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the image as the background
    if image is not None:
        ax.imshow(image)
    
    # Draw the circles
    for coords in circle_outline_list:
        ax.plot(coords[:, 0], coords[:, 1], 'b')
    
    matched_df = results_df[results_df.match_found]
    
    # TO UNCOMMENT
    for _, row in matched_df.iterrows():
        roi_points, label = row['roi_coords'], row['label']
        roi_points_array = np.array(roi_points)
        
        if not row['exactly_one_match']:
            for outline in row['matched_outlines']:
                ax.plot(outline[:, 0], outline[:, 1], 'r-')
        
        ax.plot(roi_points_array[:, 0], roi_points_array[:, 1], 'k-')
        ax.plot([roi_points_array[-1, 0], roi_points_array[0, 0]], [roi_points_array[-1, 1], roi_points_array[0, 1]], 'k-')
        
        center_x, center_y = np.mean(roi_points_array, axis=0)
        ax.text(center_x, center_y + 15, label, color='black', fontsize=10, ha='center', va='center', alpha=1, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
    unmatched_roi_points_list = results_df.loc[~results_df.match_found, 'roi_coords'].values
    
    for roi_points in unmatched_roi_points_list:
        roi_points_array = np.array(roi_points)
        ax.plot(roi_points_array[:, 0], roi_points_array[:, 1], 'r-')
        ax.plot([roi_points_array[-1, 0], roi_points_array[0, 0]], [roi_points_array[-1, 1], roi_points_array[0, 1]], 'r-')
    
    # Set aspect ratio
    ax.set_aspect('equal', 'box')
    img_name = Path(results_df.image_name[0]).stem
    fig.suptitle(img_name)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path / f"{img_name}_overall.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
