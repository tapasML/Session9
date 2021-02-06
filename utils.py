"""
This module contains utilities to display sample images.
Can be used for additional functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import dataloader
from dataloader import *
from testing import *

# display some training images in a grid
def displaysampleimage():    
    _dataiter = iter(dataloader.trainloader)
    _images, _labels = _dataiter.next()
    print('shape of images', _images.shape)
    _sample_images = _images[0:4,:,:,:] # first 4 images
   
    # show images
    __imshow__(torchvision.utils.make_grid(_sample_images))
    # print labels
    print(' '.join('%5s' % classes[_labels[j]] for j in range(4)))
    

# diaplay an image
def __imshow__(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

# display 25 misclassified images in a grid
def displaymisclassified():
    fig = plt.figure(figsize=(15,15))
    plt.title("Misclassified images\n\n")
    plt.axis("off")   
    for index in range(25):
        ax = fig.add_subplot(5, 5, index + 1, xticks=[], yticks=[])
        image = misclassifiedImages[index]   
        pred = misclassifiedPredictions[index].cpu().numpy()
        target = misclassifiedTargets[index].cpu().numpy()
        ax.imshow(image.cpu().numpy().squeeze())    
        ax.set_title(f'pred:{classes[pred]},target={classes[target]}')
        plt.subplots_adjust(wspace=1, hspace=1.5)
    plt.show()
    
# display classwise test accuracy %
def displayaccuracybyclass():
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
