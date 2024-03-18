from torch.utils.data import Dataset
import numpy as np
from skimage import io
import torch
from tqdm.auto import tqdm


from scipy.special import softmax



class SegmentationDfDataset(Dataset):
    """
    Create a dataset from a DataFrame for image segmentation tasks.

    Parameters:
    - patch_df (DataFrame): The DataFrame containing the data.
    - cv_split (int): Specifies which fold to use for cross-validation.
    - transform (callable, optional): Optional transform to be applied on a sample.
    - valid (bool): If True, use the validation set. If False, use the training set.
    - mask_col (str): Column name in DataFrame for segmentation masks.
    - patch_col (str): Column name in DataFrame for image patches.
    - instance_seg_col (str): Column name in DataFrame for instance segmentation.
    - include_patch_path (bool): If True, include the patch path in the output.
    - angio (str): Specifies the condition for including patches based on their verified angiosarcoma status.
        - 'all': Use all patches regardless of verified angiosarcoma status.
        - 'verified': Use only patches with verified angiosarcoma status.
        - 'train_all_valid_verified': For validation, use only patches with verified angiosarcoma status. For training, use all other patches.
    """

    def __init__(self, patch_df, cv_split=0, transform=None, valid=False,
                 mask_col='path_seg_one_match', patch_col='path_patch_png', instance_seg_col='path_exact_one_match',
                 include_patch_path=False, angio='all'):
        self.transform = transform
        self.patch_df = patch_df
        
        if angio == 'all':
            if valid:
                self.limited_df = patch_df[patch_df.cv == cv_split].reset_index(drop=True)
            else:
                self.limited_df = patch_df[patch_df.cv != cv_split].reset_index(drop=True)
        elif angio == 'verified':
            if valid:
                self.limited_df = patch_df[(patch_df.cv == cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
            else:
                self.limited_df = patch_df[(patch_df.cv != cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
        elif angio == 'train_all_valid_verified':
            if valid:
                self.limited_df = patch_df[(patch_df.cv == cv_split) & patch_df.verified_angiosarcoma].reset_index(drop=True)
            else:
                self.limited_df = patch_df[(patch_df.cv != cv_split) | ~patch_df.verified_angiosarcoma].reset_index(drop=True)
        
        self.masks = self.limited_df[mask_col].values
        self.patches = self.limited_df[patch_col].values
        self.instance_seg_path = self.limited_df[instance_seg_col].values
        self.include_patch_path = include_patch_path


    def __len__(self):
        return len(self.limited_df)

    def __getitem__(self, idx):
        mask = np.load(self.masks[idx])
        mask = mask.astype(int)
        patch = io.imread(self.patches[idx])
        
        # used for validation purposes. Ignored in training
        inst_seg = np.load(self.instance_seg_path[idx])
        inst_seg = torch.tensor(inst_seg)

        if self.transform:
            transformed = self.transform(image=patch, mask=mask)
            patch = transformed['image']
            mask = transformed['mask']
        if self.include_patch_path:
            return patch, mask, inst_seg, self.patches[idx]
        else:
            return patch, mask, inst_seg


########################################################################
# Functions to calculate correct cell-level instance predictions

def create_cell_masks_dict(test_inst_seg):
    cell_masks_dict = {0: 'background', 1: 'tz_pos', 2: 'tz_neg', 3: 'other_cells'}
    cell_masks_dict_str = {}

    for key in cell_masks_dict:
        if key == 0:
            cell_masks_dict_str[cell_masks_dict[key]] = [test_inst_seg[key] == 0]
        else:
            cell_masks_dict_str[cell_masks_dict[key]] = [
                test_inst_seg[key] == cell_inst
                for cell_inst in np.unique(test_inst_seg[key])
                if cell_inst != 0
            ]

    return cell_masks_dict_str


def calculate_accuracies_directly(cell_masks_dict, model_output):
    model_output_softmax = softmax(model_output, axis=0)
    metrics = {}
    predictions = []
    ground_truths = []
    cell_avg_preds = []
    intersections = []
    label_sections = []
    pred_section1s, pred_section2s, pred_section3s = [], [], []

    for key in cell_masks_dict.keys():
        correct_predictions = []

        for cell_mask in cell_masks_dict[key]:
            # Apply the cell_mask to each prediction channel
            pred_masked = [model_output_softmax[i][cell_mask] for i in range(4)]

            # Calculate the average prediction for the masked cells
            avg_pred = [np.mean(pred) for pred in pred_masked]

            pred_class = np.argmax(avg_pred)
            ground_truth_class = list(cell_masks_dict.keys()).index(key)

            correct_predictions.append(pred_class == ground_truth_class)


            pred_pixel_class = np.argmax(pred_masked, axis=0)
            intersection = np.sum(pred_pixel_class == ground_truth_class)
            # each cell_mask has one ground truth class, so the union is intersection and the length of the prediction (which is the pixel len of the mask)
            label_section = len(pred_pixel_class)
            pred_section1 = np.sum(pred_pixel_class == 1)
            pred_section2 = np.sum(pred_pixel_class == 2)
            pred_section3 = np.sum(pred_pixel_class == 3)
            # Store the prediction and ground truth class
            predictions.append(pred_class)
            ground_truths.append(ground_truth_class)
            intersections.append(intersection)
            label_sections.append(label_section)
            pred_section1s.append(pred_section1)
            pred_section2s.append(pred_section2)
            pred_section3s.append(pred_section3)
            
            # same for avg_pred
            cell_avg_preds.append(avg_pred)

        metrics[key] = correct_predictions

    metrics['predictions'] = predictions
    metrics['ground_truths'] = ground_truths
    metrics['cell_avg_preds'] = cell_avg_preds  
    metrics['intersections'] = intersections
    metrics['label_sections'] = label_sections
    metrics['pred_section1s'] = pred_section1s
    metrics['pred_section2s'] = pred_section2s
    metrics['pred_section3s'] = pred_section3s

    return metrics



def calculate_batch_accuracies(outputs, inst_seg, return_lists=False):
    batch_size = outputs.shape[0]
    metrics_list = {}

    for i in range(batch_size):
        model_output = outputs[i].cpu().detach().numpy()
        test_inst_seg = inst_seg[i].cpu().detach().numpy()

        cell_masks_dict = create_cell_masks_dict(test_inst_seg)
        metrics_dict = calculate_accuracies_directly(cell_masks_dict, model_output)

        for key, value in metrics_dict.items():
            if key not in metrics_list:
                metrics_list[key] = [np.array(value)]
            else:
                metrics_list[key].append(np.array(value))

    if return_lists:
        return {key: np.concatenate(value) for key, value in metrics_list.items()}
    else:
        average_metrics = {key: np.mean(np.concatenate(value)) for key, value in metrics_list.items()}
        return average_metrics


    
    
    
    
########################################################################
# Traning & Validation functions

def print_metrics_info(metrics_list):
    print("\nMetrics Summary:")
    print("-------------------")
    for key, values in metrics_list.items():
        # Skip 'predictions' and 'ground_truths' and cell_avg_preds since they aren't accuracy metrics
        if key in ['predictions', 'ground_truths', 'cell_avg_preds']:
            continue

        num_correct = int(np.sum(values))
        num_total = len(values)
        accuracy_percentage = num_correct / num_total * 100
        print(f"{key}: {num_correct}/{num_total} ({accuracy_percentage:.2f}%)")




def validate_unet_cell_segmentation(model, testloader, criterion, device='cuda', disable_tqdm=False):
    model.eval()
    print('Validation')

    valid_running_loss = 0.0
    counter = 0

    # Initialize the overall metrics list with the correct keys
    metrics_list = {'background': [], 'tz_pos': [], 'tz_neg': [], 'other_cells': [], 'predictions': [], 'ground_truths': [], 'cell_avg_preds': []}

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader), disable=disable_tqdm):
            counter += 1

            image, mask, inst_seg = data
            image = image.to(device)
            mask = mask.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, mask.long())
            valid_running_loss += loss.item()

            # Calculate the batch metrics list
            batch_metrics = calculate_batch_accuracies(outputs, inst_seg, return_lists=True)

            # Update the overall metrics list
            for key, value in batch_metrics.items():
                metrics_list[key].append(value)

    # Combine the overall metrics list
    for key, value in metrics_list.items():
        metrics_list[key] = np.concatenate(value)

    # Loss for the complete epoch.
    epoch_loss = valid_running_loss / counter

    return epoch_loss, metrics_list




# Training function.
def train_unet_cell_segmentation(model, trainloader, valid_loader, optimizer, criterion, device='cuda', loss_iters=10000, loss_acc_dict=None, disable_tqdm=False):
    if loss_acc_dict is None:
        loss_acc_dict = {'iters': [],'train_loss': [], 'valid_loss': [], 'train_acc': [], 'epoch_loss': [], 'epoch_acc': [], 
                         'valid_acc_background': [], 'valid_acc_tz_pos': [], 'valid_acc_tz_neg': [], 'valid_acc_other_cells': [], 
                         'valid_acc_predictions': [], 'valid_acc_ground_truths': [], 'valid_acc_cell_avg_preds': []}
    model.train()
    print('Training')
    train_running_loss = 0.0
    # train_running_correct = 0
    num_cases = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), disable=disable_tqdm):
        counter += 1
        image, mask, inst_seg = data
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, mask.long())
        train_running_loss += loss.item()
        # Calculate the accuracy
        # scalar_pred = torch.softmax(outputs.data, dim=1)[:, 1, :, :].mean(dim=[1, 2])
        # train_running_correct += ((scalar_pred > 0.5) == labels).sum().item()

        num_cases += len(outputs)
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        if ((i>0) and (i % loss_iters == 0)) or (i == len(trainloader) - 1):
            train_loss = train_running_loss / counter
            valid_epoch_loss, valid_epoch_accuracies_list = validate_unet_cell_segmentation(model, valid_loader, criterion, device=device, disable_tqdm=disable_tqdm)
            model.train()
            loss_acc_dict["train_loss"].append(train_loss)
            # loss_acc_dict["train_acc"].append(train_acc)
            loss_acc_dict["valid_loss"].append(valid_epoch_loss)
            for k, v in valid_epoch_accuracies_list.items():
                loss_acc_dict[f"valid_acc_{k}"].append(v)
            

            loss_acc_dict["iters"].append(i)
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = train_running_correct / len(trainloader.dataset)
    loss_acc_dict["epoch_loss"].append(epoch_loss)
    # loss_acc_dict["epoch_acc"].append(epoch_acc)
    
    print(f"Training loss: {epoch_loss:.3f}; Validation loss: {valid_epoch_loss:.3f}")
    print_metrics_info(valid_epoch_accuracies_list)
    return loss_acc_dict