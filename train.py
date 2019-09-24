  
import datetime
import torch
import numpy as np
import train_args

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


def load_datasets(data_dir):
     data_transforms= transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
     return datasets.ImageFolder(data_dir, transform=data_transforms)

def define_dataloaders(data):
    return torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)


def load_network(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    return model

def check_device(gpu):
    
    if not gpu:
        return torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA unavailable. Falling back to use CPU.")
        
    return device

def define_classifier(hidden_layers):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_layers[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.3)),
                          ('fc3', nn.Linear(hidden_layers[1], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier
    
def validation(model, validloader, criterion, device):
    validation_loss = accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        validation_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return validation_loss, accuracy

def train_model(arch, model, learning_rate, gpu, classifier, epochs, trainloader, validloader, train_data, save_dir):
    steps = 0
    running_loss = 0
    print_every = 20
    
    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifier
    
    criterion = nn.NLLLoss ()
    optimizer = optim.Adam (model.classifier.parameters (), lr = learning_rate)
        
    device = check_device(gpu)
    model.to(device)
    
    train_start_time = datetime.datetime.now()
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    validation_loss , accuracy = validation(model, validloader, criterion, device)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    train_end_time = datetime.datetime.now()
    train_duration = train_end_time - train_start_time
    print('Total time spent to train the model: {} mins'.format(train_duration.total_seconds()/60))  
    save_model(model, classifier, optimizer, learning_rate, train_data, arch, save_dir)
    
def save_model(model, classifier, optimizer, learning_rate, train_data, arch, save_dir):
    model.to ('cpu')
    checkpoint = {'model_state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'classifier': classifier,
                  'optimizer_state': optimizer.state_dict(),
                  'learning_rate': learning_rate,
                  'arch': arch}

    torch.save(checkpoint, save_dir)
    
def main():
    args = train_args.get_args()
    print(args)
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    
    train_data = load_datasets(train_dir)
    valid_data = load_datasets(valid_dir)
    
    trainloader = define_dataloaders(train_data)
    validloader = define_dataloaders(valid_data)
    inputs, labels = next(iter(trainloader))
    print(inputs.size())
    
    hidden_layers = args.hidden_layers.split(',')
    hidden_layers = [int(i) for i in hidden_layers]
    
    classifier = define_classifier(hidden_layers)
    print("Loading pre-trained network")
    model = load_network(args.arch)
    print(model)
    train_model(args.arch, 
                model, 
                args.learning_rate, 
                args.gpu, 
                classifier, 
                args.epochs, 
                trainloader, 
                validloader, 
                train_data,
                args.save_dir)

    
if __name__ == '__main__':
    main()
    
