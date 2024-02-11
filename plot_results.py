#%%
import os
import torch
import matplotlib.pyplot as plt


def load_results(directory):

    epochs = []
    training_loss = []
    valid_loss = []
    training_accuracy = []
    valid_accuracy = []

    for file in os.listdir(directory):
        if file.endswith('.pth'):
            filepath = os.path.join(directory, file)
            checkpoint = torch.load(filepath)
            epochs.append(checkpoint['epoch'])
            training_loss.append(checkpoint['train_loss'])
            valid_loss.append(checkpoint['valid_loss'])
            training_accuracy.append(checkpoint['train_acc'])
            valid_accuracy.append(checkpoint['valid_acc'])

        else:
            print(f'File {file} does not correspond')

    combined = list(zip(epochs, training_loss, valid_loss, training_accuracy, valid_accuracy))
    combined_sorted = sorted(combined, key = lambda x: x[0])

    epochs, training_loss, valid_loss, training_accuracy, valid_accuracy = zip(*combined_sorted)

    return epochs, training_loss, valid_loss, training_accuracy, valid_accuracy

def plot_results(epochs, training_loss, valid_loss, training_accuracy, valid_accuracy):
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].plot(epochs, training_loss, label='Training Loss', color='b')
    ax[0].plot(epochs, valid_loss, label='Validation Loss', color='r')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, training_accuracy, label='Training Accuracy', color='b')
    ax[1].plot(epochs, valid_accuracy, label='Validation Accuracy', color='r')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    epochs3, train_loss3, val_loss3, train_acc3, val_acc3 = load_results('pretrained_model')
    epochs, train_loss, val_loss, train_acc, val_acc = load_results('model')
    
    plot_results(epochs3, train_loss3, val_loss3, train_acc3, val_acc3)
    plot_results(epochs, train_loss, val_loss, train_acc, val_acc)




# %%
