'''
change directory to ImageClassifier
sample run 1: python train.py (all defaults included)
sample run 2: python train.py --arch "densenet121" --epochs 5
'''

import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from collections import OrderedDict
from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet201', choices=['densenet201', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--scheduler', dest='scheduler', default='StepLR')
    parser.add_argument('--hidden_layers', dest='hidden_layers', default='512')
    parser.add_argument('--epochs', dest='epochs', default='6')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 40
    count_time = time.time()
    
    for e in range(epochs):
        start = time.time()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1 

            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                model.cpu()
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(dataloaders[1]):
                    
                        optimizer.zero_grad()

                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                            model.to('cuda:0')
                        else:
                            pass 
                        
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valloss = valloss / len(dataloaders[1])
                accuracy = accuracy /len(dataloaders[1])
                duration = time.time() - start
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Duration: {:.1f} s | ".format(duration),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Loss {:.3f}".format(valloss),
                      "Accuracy: {:.3f}".format(accuracy),
                     )

                running_loss = 0
    end_time = time.time()
    total_time = end_time - count_time
    print(" Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))
            
def main():
    print("Initiating...")
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])]) 

    testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(val_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    input_size = model.classifier.in_features
    hidden_layers = int(args.hidden_layers)
    output_size = 102
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_layers)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers, output_size)),
                              ('output', nn.LogSoftmax(dim=1))]))


    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()