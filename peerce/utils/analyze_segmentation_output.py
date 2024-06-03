import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import cv2

from PIL import Image, ImageDraw, ImageFont
from scipy.special import softmax

from peerce.deep_learning.train_unet_cell_segmentation_and_valid import calculate_accuracies_directly
####################################################################################################
# mostly analysis functions to create dicts with information about cell masks and predictions

def create_cell_masks_dict(test_inst_seg):
    cell_masks_dict = {0: 'background', 1: 'tz_pos', 2: 'tz_neg', 3: 'other_cells'}
    cell_masks_dict_str = {}

    for key in cell_masks_dict:
        cell_masks = []
        if key == 0:
            cell_mask = test_inst_seg[key] == 0
            cell_masks.append(cell_mask)
        else:
            for cell_inst in np.unique(test_inst_seg[key]):
                if cell_inst == 0:
                    continue
                else:
                    cell_mask = test_inst_seg[key] == cell_inst
                    cell_masks.append(cell_mask)
        
        cell_masks_dict_str[cell_masks_dict[key]] = cell_masks

    return cell_masks_dict_str



def get_cell_prediction(model_output, cell_mask):
    # Apply softmax to model_output
    model_output_softmax = softmax(model_output, axis=0)

    # Extract prediction channels
    background_pred = model_output_softmax[0]
    tz_pos_pred = model_output_softmax[1]
    tz_neg_pred = model_output_softmax[2]
    other_cells_pred = model_output_softmax[3]

    # Apply the cell_mask to each prediction channel
    background_pred_masked = background_pred[cell_mask]
    tz_pos_pred_masked = tz_pos_pred[cell_mask]
    tz_neg_pred_masked = tz_neg_pred[cell_mask]
    other_cells_pred_masked = other_cells_pred[cell_mask]

    # Calculate the average prediction for the masked cells
    background_avg_pred = np.mean(background_pred_masked)
    tz_pos_avg_pred = np.mean(tz_pos_pred_masked)
    tz_neg_avg_pred = np.mean(tz_neg_pred_masked)
    other_cells_avg_pred = np.mean(other_cells_pred_masked)

    # Return the prediction vector
    return np.array([background_avg_pred, tz_pos_avg_pred, tz_neg_avg_pred, other_cells_avg_pred])


def add_prediction_to_dict(cell_masks_dict, model_output):
    prediction_dict = cell_masks_dict.copy()
    for key in cell_masks_dict.keys():
        predictions = []
        for cell_mask in cell_masks_dict[key]:
            prediction = get_cell_prediction(model_output, cell_mask)
            predictions.append(prediction)
        prediction_dict[key + "_prediction"] = predictions
    return prediction_dict


def calculate_accuracies(prediction_dict):
    accuracies = {}
    for key in prediction_dict.keys():
        if not key.endswith("_prediction"):
            continue

        ground_truth_key = key[:-11]  # Remove '_prediction' from the key to get the ground truth key
        correct_predictions = []

        for prediction in prediction_dict[key]:
            pred_class = np.argmax(prediction)
            if ground_truth_key == 'tz_pos' and pred_class == 1:
                correct_predictions.append(True)
            elif ground_truth_key == 'tz_neg' and pred_class == 2:
                correct_predictions.append(True)
            elif ground_truth_key == 'other_cells' and pred_class == 3:
                correct_predictions.append(True)
            elif ground_truth_key == 'background' and pred_class == 0:
                correct_predictions.append(True)
            else:
                correct_predictions.append(False)

        accuracies[ground_truth_key] = correct_predictions

    return accuracies


# # Example usage
# cell_masks_dict = create_cell_masks_dict(inst_seg)
# prediction_dict = add_prediction_to_dict(cell_masks_dict, model_out)
# accuracies_dict = calculate_accuracies(prediction_dict)

# # print(f"Prediction dict: {prediction_dict}")
# print(f"Accuracies dict: {accuracies_dict}")


def calculate_batch_accuracies(outputs, inst_seg, return_lists=False):
    batch_size = outputs.shape[0]
    accuracies_list = {}
    for i in range(batch_size):
        model_output = outputs[i].cpu().detach().numpy()
        test_inst_seg = inst_seg[i].cpu().detach().numpy()

        cell_masks_dict = create_cell_masks_dict(test_inst_seg)
        prediction_dict = add_prediction_to_dict(cell_masks_dict, model_output)
        accuracies_dict = calculate_accuracies(prediction_dict)

        for key, value in accuracies_dict.items():
            if key not in accuracies_list:
                accuracies_list[key] = [np.array(value, dtype=bool)]
            else:
                accuracies_list[key].append(np.array(value, dtype=bool))

    if return_lists:
        return {key: np.concatenate(value) for key, value in accuracies_list.items()}
    else:
        average_accuracies = {key: np.mean(np.concatenate(value)) for key, value in accuracies_list.items()}
        return average_accuracies

# # Example usage
# batch_accuracies = calculate_batch_accuracies(outputs, inst_seg, return_lists=False)
# print(f"Batch accuracies: {batch_accuracies}")

# batch_accuracies_list = calculate_batch_accuracies(outputs, inst_seg, return_lists=True)
# print(f"Batch accuracies list: {batch_accuracies_list}")


def get_description_string(inst_seg, model_out):
    cell_masks_dict = create_cell_masks_dict(inst_seg)
    acc_dict = calculate_accuracies_directly(cell_masks_dict, model_out)

    fstr_list = ["Accuracies for labeled cell instances:"]
    for k, v in acc_dict.items():
        if k in ['background', 'intersections', 'label_sections', 'pred_section1s', 'pred_section2s', 'pred_section3s',
                 'predictions', 'ground_truths', 'cell_avg_preds']:
            continue
        if len(v) > 0:
            fstr = f"{k}: {np.mean(v):.2%}"
        else:
            fstr = f"{k}: no instances"
        fstr_list.append(fstr)
        
    description_string = '\n'.join(fstr_list)
    return description_string


####################################################################################################
# Visualization functions

def visualize_cell_instance_predictions(inst_seg, prediction_dict, accuracies_dict, outpath=None, mode='viz'):
    # Create a blank image to draw on
    inst_seg_np = inst_seg.detach().cpu().numpy()
    image_height, image_width = inst_seg_np.shape[-2:]
    output_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    output_image = output_image + 100  # gray for non-annotated cells
    
    # Iterate through each cell class
    for cell_class, cell_masks in prediction_dict.items():
        if cell_class.endswith("_prediction"):
            ground_truth_key = cell_class[:-11]  # Remove '_prediction' from the key to get the ground truth key
            for idx, (cell_mask, cell_prediction) in enumerate(zip(prediction_dict[ground_truth_key], cell_masks)):
                # Color the cell according to its predicted class
                pred_class = np.argmax(cell_prediction)
                correct = accuracies_dict[ground_truth_key][idx]
                if ground_truth_key == 'background':
                    color = (128, 128, 128)  # gray for background
                elif pred_class == 0 and not correct:
                    color = (255, 255, 255) # white for cell as background
                elif pred_class == 1:
                    color = (255, 0, 0)  # red for tz_pos
                elif pred_class == 2:
                    color = (0, 0, 255)  # blue for tz_neg
                else:
                    color = (255, 255, 0)  # yellow for other_cells
                
                output_image[cell_mask] = color
                
                # if ground_truth_key == 'background', then we don't want to draw the correct/incorrect annotation
                if ground_truth_key == 'background':
                    continue
                
                # Annotate whether the prediction was correct
                y, x = np.argwhere(cell_mask).float().mean(axis=1)
                # x += 10
                # y += 10
                correct = accuracies_dict[ground_truth_key][idx]
                annotation = "C" if correct else "I"
                
                # Convert the OpenCV image to a PIL image
                pil_image = Image.fromarray(output_image)

                # Set the annotation text and color based on the prediction correctness
                if correct:
                    annotation = "v"
                    annotation_color = (0, 255, 0)  # Green for correct
                else:
                    annotation = "X"
                    annotation_color = (255, 0, 0)  # Red for incorrect

                # Draw the annotation text using PIL
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype("arial.ttf", 20)
                # font = ImageFont.load_default()
                draw.text((int(x), int(y)), annotation, font=font, fill=annotation_color)

                # Convert the PIL image back to an OpenCV image
                output_image = np.array(pil_image)

    if outpath is None:
        if mode == 'viz':
            plt.imshow(output_image)
            # plt.axis("off")
            plt.show()
        elif mode == 'return':
            return output_image
    else:
        plt.imsave(outpath, output_image)



def visualize_all_cell_predictions(model_out, inst_seg, outpath=None, mode='viz'):
    # Do softmax on the model output
    model_out_softmax = softmax(model_out, axis=0)

    # Create a blank image to draw on
    inst_seg_np = inst_seg.numpy()[0] # channel 0 has all the cell instances
    image_height, image_width = inst_seg_np.shape[-2:]
    output_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    output_image = output_image + 128  # gray for non-annotated cells

    # Iterate through each cell instance
    for cell_value in np.unique(inst_seg_np):
        if cell_value == 0:
            continue

        cell_mask = inst_seg_np == cell_value
        cell_preds = [np.mean(channel[cell_mask]) for channel in model_out_softmax]
        cell_pred = np.argmax(cell_preds)

        # Color the cell according to its predicted class
        if cell_pred == 0:
            color = (255, 255, 255)  # white for background
        elif cell_pred == 1:
            color = (255, 0, 0)  # red for tz_pos
        elif cell_pred == 2:
            color = (0, 0, 255)  # blue for tz_neg
        else:
            color = (255, 255, 0)  # yellow for other_cells

        output_image[cell_mask] = color

    if outpath is None:
        if mode == 'viz':
            plt.imshow(output_image)
            # plt.axis("off")
            plt.show()
        elif mode == 'return':
            return output_image
    else:
        plt.imsave(outpath, output_image)
        
        
# just a little helper :)
def get_cell_instance_predictions_roi_viz(model_out, inst_seg):
    cell_masks_dict = create_cell_masks_dict(inst_seg)
    prediction_dict = add_prediction_to_dict(cell_masks_dict, model_out)
    accuracies_dict = calculate_accuracies(prediction_dict)
    cell_instance_predictions_roi_viz = visualize_cell_instance_predictions(inst_seg, prediction_dict, accuracies_dict, mode='return')
    return cell_instance_predictions_roi_viz



####################################################################################################
# Basically a all in one function, combining many of the above functions

def visualize_image_and_model_output(true_img, model_output, inst_seg, info, outpath=None):
  
    # Get the class with the maximum prediction value for each pixel
    predicted_class = np.argmax(model_output, axis=0)
    
    # Squeeze the extra dimensions
    # predicted_class = np.squeeze(predicted_class)
    
    # Create a subplot with two images side by side
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Display the original image on the left
    axes = axes.flatten()
    axes[0].imshow(true_img)
    axes[0].set_title('Original Image')
    
    # Display the model output on the right using the visualize_unet_segmentation_array function
    cmap = ListedColormap(['gray', 'white', 'red', 'blue', 'yellow'])
    mapped_array = predicted_class.copy()
    mapped_array[predicted_class == 0] = 1
    mapped_array[predicted_class == 1] = 2
    mapped_array[predicted_class == 2] = 3
    mapped_array[predicted_class == 3] = 4
    axes[1].imshow(mapped_array, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    axes[1].set_title('Model Output (Background: White, \nTZ+: Red, TZ-: Blue, Other Cell: Yellow)')
    
    
    roi_viz = get_cell_instance_predictions_roi_viz(model_output, inst_seg)
    all_cell_viz = visualize_all_cell_predictions(model_output, inst_seg, mode='return')
    
    axes[2].imshow(roi_viz)
    axes[2].set_title('Cell Instance Predictions for given Cellpose ROIs \n (Background: White, TZ+: Red, TZ-: Blue, Other Cell: Yellow)')
    
    axes[3].imshow(all_cell_viz)
    axes[3].set_title('All Cell Instance Predictions')
    
    fig.suptitle(info, fontsize=16)
    
    # Add a figure description text at the bottom of the subplot figure
    description = get_description_string(inst_seg, model_output)
    fig.text(0.5, -0.07, description, ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # Save or show the visualization
    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
        
####################################################################################################
# A nice visualization of the annotated input image



def plot_circles_and_roi_points(true_img, inst_seg, info_ground_truth, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the true_img as the background
    ax.imshow(true_img)
    inst_seg = np.array(inst_seg)
    
    # Define the cell types
    cell_types = {1: 'tz_pos', 2: 'tz_neg', 3: 'other_cells'}

    # Iterate over channels in inst_seg
    for idx, cell_type in cell_types.items():
        channel = inst_seg[idx]
        unique_cells = np.unique(channel)

        for cell_value in unique_cells:
            if cell_value == 0:
                continue

            # Get the cell mask and find its boundary points
            cell_mask = (channel == cell_value)
            contours, _ = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                boundary_points = contour.squeeze()
                
                # Add the first point to the end of the contour to ensure it's closed
                closed_boundary_points = np.concatenate((boundary_points, boundary_points[0:1]), axis=0)
                ax.plot(closed_boundary_points[:, 0], closed_boundary_points[:, 1], 'r-')

                # Calculate the center of the cell
                center_y, center_x = np.argwhere(cell_mask).mean(axis=0)

                ax.text(center_x, center_y + 15, f"{cell_type}", color='black', fontsize=10, ha='center', va='center', alpha=1, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # Set aspect ratio
    ax.set_aspect('equal', 'box')
    # img_name = "Image_Name"
    fig.suptitle(info_ground_truth)
    plt.tight_layout()

    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

