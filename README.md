# A Comparative Study of CNNs in Bird Identification
## Description
This repository contains a deep learning project focused on classifying bird species using convolutional neural networks (CNN). 
It features a comparison between a custom-built CNN and the pretrained EfficientNetB0 model, emphasizing their performance through F1-scores. 
This project demonstrates the practical application of convolutional neural networks in image recognition, leveraging PyTorch for model development and training.

## Dataset
The models were trained and evaluated using a dataset sourced from Kaggle (https://www.kaggle.com/datasets/gpiosenka/100-bird-species), specifically designed for image classification tasks and includes a diverse collection of high-quality images (224x224x3) spanning 525 different bird species.

### Data Preproccesing
The dataset was first loaded using the 'ImageFolder' class, which automatically labels all images based on file names and facilitates an organized structure for model training.
Several transformations were applied to the dataset to optimize model performance and generalization, these included resizing to maintain consistency and a combination of augmentation techniques to introduce variability.

```
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomApply([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(),
            ], p=0.5),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'test':transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    }
```

The data is then divided into training, validation and test sets with DataLoader instances to allow batch processing and data shuffling for the training and validation phases, and orderly processing in the test phase.

```
    train_set = CustomDataset(train_subset, transform=data_transforms['train'])
    val_set = CustomDataset(val_subset, transform=data_transforms['valid'])
    test_set = CustomDataset(test_subset, transform=data_transforms['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
```

## Models
### Custom CNN Model (BirdCNN)
The custom CNN Model, features a sequential architecture that includes multiple convolutional layers that escalate in complexity and depth. 
We incorporate ReLU acitvation for non-linearity, dropout for regularization and batch normalization to accelerate training.

```
class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout1 = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm2d(32) 

        self.conv3 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout3 = nn.Dropout(p=0.1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout4 = nn.Dropout(p=0.1)
        self.bn4 = nn.BatchNorm2d(128) 

        self.fc1 = nn.Linear(128*7*7,512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512,num_classes)
```

### Pretrained CNN Model (EfficientNetB0)
EfficientNetB0 is acclaimed for its efficiency in classifying over 1000 different object categories, proving to be suitable for comparison with the custom CNN. We freeze the original model's weights and add a custom classifier (top layer) comprising a linear layer, ReLU activation, batch normalization and dropout. This allows for fine-tuning, as we do not modify the weights of any layers other than the newly added top layer 

```
model = models.efficientnet_b0(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Linear(in_features=num_features,
              out_features=512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(in_features=512, out_features=525)
)
```

## Training results
The summary of both models hyperparameters is shown as follows:\
**Batch size**: 128\
**Epochs**: 45\
**Input Shape**: (224,224,3)\
**Output layer**: 525

### BirdCNN Learning Curve

![BIRDCNN](https://github.com/stearsail/Bird_classifier/assets/129506811/3d076bb7-b230-43af-b733-ae6288319175)

### EfficientNetB0 Learning Curve

![EFFICIENTNET](https://github.com/stearsail/Bird_classifier/assets/129506811/39452ea5-7d3d-48fa-b872-2c662bb91559)

### Model evaluation
The test dataset is used to evaluate the performance of both models. For an accurate comparison we compute the F1-score (harmonic mean of precision and recall).

```
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
    print("F1-Score (BirdCNN Model):", f1_model)
```
**Output**:\
F1-Score (Pretrained Model): 0.9586968027622187
F1-Score (BirdCNN Model): 0.9522995276329483
