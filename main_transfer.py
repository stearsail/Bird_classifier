import os
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
from efficientnet_cnn import model
from dataset import create_dataloaders, load_dataset, CURRENT_DIR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, optimizer, criterion, train_loader, epoch_index):
    running_loss = 0.
    total_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # print(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        
        total_correct = 0
        total_instances = 0

        predictions = torch.argmax(outputs, dim=1)
        correct_predictions = (predictions == labels).sum().item()
        total_correct += correct_predictions
        total_instances += labels.size(0)

        taccuracy = round(total_correct / total_instances, 3)

        total_loss += loss.item()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('EPOCH: {} BATCH: {} LOSS: {:.5f}'.format(epoch_index+1,i+1,last_loss))
        running_loss = 0.
    
    return total_loss/len(train_loader), taccuracy

def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, epochs):
    best_val_loss = float('inf')
    try:
        for epoch in range(epochs):

            model.train(True)
            learning_rate = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {learning_rate}')
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, epoch)

            model.eval()
            total_vloss = 0.
            total_correct = 0
            total_instances = 0

            with torch.no_grad():
                for _, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    total_vloss += vloss.item()
                    predictions = torch.argmax(model(vinputs), dim=1)
                    correct_predictions = (predictions == vlabels).sum().item()
                    total_correct += correct_predictions
                    total_instances += vlabels.size(0)
                
                valid_acc = round(total_correct / total_instances, 3)
                val_loss = total_vloss/len(val_loader)
                scheduler.step(val_loss)


            print(f'EPOCH {epoch+1} - AVG TRAIN-LOSS {train_loss:.5f} - AVG VALID-LOSS {val_loss:.5f}')
            print('TRAINING ACCURACY: {:.5f}%'.format(train_acc*100))
            print('VALIDATION ACCURACY: {:.5f}%'.format(valid_acc*100))

            if val_loss < best_val_loss:
                
                best_val_loss = val_loss

                if not os.path.exists(f'model/'):
                    os.mkdir(f'model/')

                print('Saving model with improved validation loss...')

                data_to_save={
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': best_val_loss,
                    'train_loss': train_loss,
                    'valid_acc': valid_acc,
                    'train_acc': train_acc
                }
                torch.save(data_to_save, f'pretrained_model/Model_epoch{epoch+1}.pth')
    except KeyboardInterrupt:
        pass
    return best_val_loss

if __name__ == "__main__":


    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr = 0.001)
    
    criterion = nn.CrossEntropyLoss().to(device)

    scheduler = lrs.ReduceLROnPlateau(optimizer = optimizer,
                                      mode = 'min',
                                      factor = 0.1,
                                      patience = 5,
                                      threshold = 1e-2) #Default threshold is 1e-4
    
    dataset = load_dataset(os.path.join(CURRENT_DIR, 'Data', 'full_dataset'))

    train_loader, val_loader, test_loader = create_dataloaders(dataset = dataset,
                                                               test_size=0.1,
                                                               val_size=0.1,
                                                               batch_size=128)


    vloss_list, tloss_list, accuracy_list, vaccuracy_list = train_model(model = model,
                                                                        optimizer = optimizer,
                                                                        criterion = criterion,
                                                                        scheduler = scheduler,
                                                                        train_loader = train_loader,
                                                                        val_loader = val_loader,
                                                                        epochs = 100)

