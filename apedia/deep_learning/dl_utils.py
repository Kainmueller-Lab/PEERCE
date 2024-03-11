##########################################################
# Start of cell 0fdbcd58
##########################################################

import torch
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
##########################################################
# End of cell 0fdbcd58
##########################################################

##########################################################
# Start of cell b9b11010
##########################################################

def save_model(epochs, model, model_name, optimizer, criterion, out_path, info):
    """
    Function to save the trained model to disk.
    'model_name' should be the name of the model as named within the torch models module.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'model_name': model_name
                }, out_path / f"model_{info}.pth")

    
def save_plots(train_acc, valid_acc, train_loss, valid_loss, out_path, info, x_label="Epochs"):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validation accuracy'
    )
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out_path / f"validation_accuracy_{info}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validation loss'
    )
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_path / f"loss_{info}.png")
##########################################################
# End of cell b9b11010
##########################################################