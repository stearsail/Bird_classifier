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

`    data_transforms = {
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
    }`

The data is then divided into training, validation and test sets with DataLoader instances to allow batch processing and data shuffling for the training and validation phases, and orderly processing in the test phase.

`  train_set = CustomDataset(train_subset, transform=data_transforms['train'])
    val_set = CustomDataset(val_subset, transform=data_transforms['valid'])
    test_set = CustomDataset(test_subset, transform=data_transforms['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
`

