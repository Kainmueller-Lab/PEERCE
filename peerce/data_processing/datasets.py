from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Required constants.
IMAGE_SIZE = 512 # Image size of resize when applying transforms.
BATCH_SIZE = 16 
NUM_WORKERS = 4 # Number of parallel processes for data preparation.
##########################################################
# End of cell 0fdbcd58
##########################################################

##########################################################
# Start of cell b9b11010
##########################################################

def get_datasets(dataset_path, pretrained=True, train_transform=None):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    
    if train_transform is None:
        train_transform = get_train_transform
    dataset_train = datasets.ImageFolder(
        dataset_path / 'train_set', 
        transform=(train_transform(IMAGE_SIZE, pretrained))
    )
    dataset_valid = datasets.ImageFolder(
        dataset_path / 'validation_set', 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
#     dataset_size = len(dataset)
#     # Calculate the validation dataset size.
#     valid_size = int(VALID_SPLIT*dataset_size)
#     # Radomize the data indices.
#     indices = torch.randperm(len(dataset)).tolist()
#     # Training and validation sets.
#     dataset_train = Subset(dataset, indices[:-valid_size])
#     dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers=4):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader 
##########################################################
# End of cell b9b11010
##########################################################

##########################################################
# Start of cell de952fa5
##########################################################

# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained=True):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
#         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#         transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Training transforms with more augmentations
def get_more_augmented_train_transform(IMAGE_SIZE, pretrained=True):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(20,20)),
        # transforms.RandomAffine(degrees=(0, 360), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(40,40)),
        transforms.ColorJitter(brightness=.2, contrast=0.2, hue=0.1, saturation=0.2),
        # transforms.ColorJitter(brightness=.4, contrast=0.4, hue=0.2, saturation=0.4),
        # transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform



# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained=True):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform


def get_more_augmented_train_transform_6chan(IMAGE_SIZE, pretrained=True, n_imgs=1):
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(20,20)),
        # transforms.RandomAffine(degrees=(0, 360), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(40,40)),
        # transforms.ColorJitter(brightness=.2, contrast=0.2, hue=0.1, saturation=0.2),
        # transforms.ColorJitter(brightness=.4, contrast=0.4, hue=0.2, saturation=0.4),
        # transforms.ToTensor(),
        normalize_transform(pretrained, n_imgs=n_imgs)
    ])
    return train_transform


def get_valid_transform_6chan(IMAGE_SIZE, pretrained=True, n_imgs=1):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor(),
        normalize_transform(pretrained, n_imgs=n_imgs)
    ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained, n_imgs=1):
    # updated to also normalize tensors with >3 channels
    if pretrained: # Normalization for pre-trained weights.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(n_imgs-1):
            mean.extend(mean)
            std.extend(std)
        normalize = transforms.Normalize(
            mean=mean,
            std=std
            )
    else: # Normalization when training from scratch.
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for i in range(n_imgs-1):
            mean.extend(mean)
            std.extend(std)
        normalize = transforms.Normalize(
            mean=mean,
            std=std
        )
    return normalize
##########################################################
# End of cell de952fa5
##########################################################

##########################################################
# Start of cell 930591f2
##########################################################

def show_tensor(tensor, normalized=False):
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                               transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]) ])

    if normalized:
        tensor = invTrans(tensor)
    tensor = tensor.permute(1,2,0)
    plt.imshow(tensor)
    plt.show()
##########################################################
# End of cell 930591f2
##########################################################