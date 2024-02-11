#%%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dataset import create_dataloaders, load_dataset, CURRENT_DIR
from efficientnet_cnn import model as pretrained_model
from cnn import BirdCNN

def load_model(file_path, model):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":

    model = BirdCNN(525).to(device='cuda')

    # pretrained_model = pretrained_model.to(device='cuda')

    dataset = load_dataset(os.path.join(CURRENT_DIR, 'Data', 'full_dataset'))

    train_loader, val_loader, test_loader = create_dataloaders(dataset = dataset,
                                                               test_size=0.1,
                                                               val_size=0.1,
                                                               batch_size=128)

    model = load_model('model/Model_epoch45.pth', model)
    pretrained_model = load_model('pretrained_model/Model_epoch75.pth', pretrained_model)
    
    model.eval()
    pretrained_model.eval()

    true_labels = []
    preds_pretrained = []
    preds_model = []

    for inputs, labels in test_loader:
        
        inputs = inputs.to(device='cuda')
        labels = labels.to(device='cuda')

        outputs_pretrained = torch.softmax(pretrained_model(inputs), dim = 1)
        outpouts_model = torch.softmax(model(inputs), dim = 1)

        _, predicted_pretrained = torch.max(outputs_pretrained, 1)
        _, predicted_model = torch.max(outpouts_model, 1)

        true_labels.extend(labels.cpu().numpy())
        preds_pretrained.extend(predicted_pretrained.cpu().numpy())
        preds_model.extend(predicted_model.cpu().numpy())

    true_labels = np.array(true_labels)
    preds_pretrained = np.array(preds_pretrained)
    preds_model = np.array(preds_model)

    f1_pretrained = f1_score(true_labels, preds_pretrained, average='weighted')
    f1_model = f1_score(true_labels, preds_model, average='weighted')

    print("F1-Score (Pretrained Model):", f1_pretrained)
    print("F1-Score (Your Model):", f1_model)

