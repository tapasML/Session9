"""
This is the contains functions to train the model
"""
import torch.optim as optim                          
from torchsummary import summary 

import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = 0
optimizer = 0
scheduler = 0

# create the Resnet18 model
def createmodel():
    global model
    model = ResNet34().to(device)
    return model

# generate model summary for analysis
def modelSummary(model):    
    summary(model, input_size=(3, 32, 32))

# define loss functions etc
def definelossfunction():
    global criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = LR, momentum = MOMENTUM)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)

# train the model. Note that single channel image is trained
def trainmodel():    
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader.trainloader, 0):
            # get the inputs
            inputs, labels = data
            #inputs = inputs[:,0:1,:,:] #grayscale
            inputs = inputs.to(device)
            labels = labels.to(device)
            if i == 0:
                print(inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 40 == 39:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        scheduler.step()
    print('Finished Training')
    return model