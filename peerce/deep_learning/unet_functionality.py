import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm.auto import tqdm



# model = smp.Unet(
#     encoder_name="timm-efficientnet-b5", # "resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,                      # model output channels (number of classes in your dataset)
#     activation=None #'sigmoid'               
# )



# model = smp.UnetPlusPlus(
#     encoder_name="timm-efficientnet-b5", # "resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,                      # model output channels (number of classes in your dataset)
#     activation='sigmoid'               
# )

class NInputChanUnet(nn.Module):
    
    def __init__(self, n_channels_in=3, model_cls=None, *args, **kwargs):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.model = self.get_model(model_cls, *args, **kwargs)
        
        
    def get_model(self, model_cls, *args, **kwargs):
        if model_cls is None:
            model_cls = smp.Unet
        model = model_cls(*args, **kwargs)
        if self.n_channels_in == 6:
            old_input_layer = model.encoder.conv_stem
            new_input_layer = nn.Conv2d(6, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

            new_input_layer.weight.data[:, :3, :, :] = old_input_layer.weight.data/2
            new_input_layer.weight.data[:, 3:, :, :] = old_input_layer.weight.data/2

            model.encoder.conv_stem = new_input_layer
        return model
    
    
    def forward(self, x):
        return self.model(x)
                
                
    def predict_segmentation(self, x):
        with torch.no_grad():
            pred = self.model(x.unsqueeze(0).cuda())
            pred = torch.softmax(pred, dim=1).squeeze().cpu().detach().numpy()[1]
            return pred







def validate_unet(model, testloader, criterion, device='cuda', disable_tqdm=False):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader), disable=disable_tqdm):
            counter += 1
            
            image, labels, mask = data
            image = image.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, mask.long())
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            scalar_pred = torch.softmax(outputs.data, dim=1)[:, 1, :, :].mean(dim=[1, 2])
            valid_running_correct += ((scalar_pred > 0.5) == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc =  valid_running_correct / len(testloader.dataset)
    return epoch_loss, epoch_acc


# Training function.
def train_more_valid_unet(model, trainloader, valid_loader, optimizer, criterion, device='cuda', loss_iters=10000, loss_acc_dict=None, disable_tqdm=False):
    if loss_acc_dict is None:
        loss_acc_dict = {'iters': [],'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'epoch_loss': [], 'epoch_acc': []}
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    num_cases = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), disable=disable_tqdm):
        counter += 1
        image, labels, mask = data
        image = image.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, mask.long())
        train_running_loss += loss.item()
        # Calculate the accuracy
        scalar_pred = torch.softmax(outputs.data, dim=1)[:, 1, :, :].mean(dim=[1, 2])
        train_running_correct += ((scalar_pred > 0.5) == labels).sum().item()

        num_cases += len(scalar_pred)
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        if i % loss_iters == 0:
            train_loss = train_running_loss / counter
            train_acc = (train_running_correct / num_cases)
            train_running_loss = 0.0
            train_running_correct = 0
            num_cases = 0
            counter = 0
            
            valid_epoch_loss, valid_epoch_acc = validate_unet(model, valid_loader, criterion, device=device, disable_tqdm=disable_tqdm)
            model.train()
            loss_acc_dict["train_loss"].append(train_loss)
            loss_acc_dict["train_acc"].append(train_acc)
            loss_acc_dict["valid_loss"].append(valid_epoch_loss)
            loss_acc_dict["valid_acc"].append(valid_epoch_acc)
            

            loss_acc_dict["iters"].append(i)
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_correct / len(trainloader.dataset)
    loss_acc_dict["epoch_loss"].append(epoch_loss)
    loss_acc_dict["epoch_acc"].append(epoch_acc)
    
    print(f"Training loss: {epoch_loss:.3f}; training accuracy: {epoch_acc:.2%}")
    print(f"Validation loss: {valid_epoch_loss:.3f}; validation accuracy: {valid_epoch_acc:.2%}")
    return loss_acc_dict