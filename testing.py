"""
This module contains functions to test the model
"""
import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import training
from training import *

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

train_losses = []
test_losses = []

misclassifiedImages = []      # store mis-classified images 
misclassifiedPredictions = [] # wrong predictions for mis-classified images
misclassifiedTargets = []     # correct targets for mis-classified images

class_correct = list(0. for i in range(10))  
class_total = list(0. for i in range(10))   # classwise accuracy


# test the network with test data
def testmodel(trainedmodel):
    with torch.no_grad():
        test_loss = 0   
        totalMismatch = 0
        correct = 0
        total = 0
        for data in dataloader.testloader:
            original_images, labels = data
            #images = original_images[:,0:1,:,:]
            images = original_images
            images=images.to(device)
            labels=labels.to(device)
            output = trainedmodel(images)
            
            add_stats_classified(output, labels) # add to classification stats
            
            '''_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()'''
            total += labels.size(0)
            test_loss += torch.nn.functional.nll_loss(output, labels, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
            # find mis-classified images, predictions, targets
            result = pred.squeeze() == labels
            indices = [i for i, element in enumerate(result) if not element]           
            totalMismatch += len(indices)
            for i in indices:
                misclassifiedImages.append(np.transpose(original_images[i],(1,2,0)) .squeeze())
                misclassifiedPredictions.append(pred.squeeze()[i])
                misclassifiedTargets.append(labels[i])   
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
   

# Displays what % of classes are classified/misclassified
def add_stats_classified(output, labels): 
    _, predicted = torch.max(output, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
    