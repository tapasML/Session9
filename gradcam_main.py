#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import training
from training import *

from grad_cam import (
    BackPropagation,    
    GradCAM,  
)


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    print('loaded images from path {}'.format(image_paths))
    return images, raw_images


def getclasses():    
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    #I am using 32X32 image
    raw_image = cv2.resize(raw_image, (32,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def gradcam_classify():
    image_paths = ['samples/cifar-grad.png']
    target_layer = 'layer3'
    topk = 1
    output_dir = 'results'
  
    classes = getclasses()
  
    model.to(device)
    model.eval()

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    
    # =========================================================================   

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted    

    # =========================================================================
    print("Grad-CAM in action:")
    gcam = GradCAM(model=model)    
    _ = gcam.forward(images)
    
    for i in range(topk):       
        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):            
            # Grad-CAM
            global filename
            filename=osp.join(output_dir, "{}-{}-gradcam-{}-{}.png".format(j, 'resnet', target_layer, classes[ids[j, i]] ),)
                
            save_gradcam(                
                filename=filename,
                gcam=regions[j, 0],
                raw_image=raw_images[j],               
            ) 
            print("Grad am file generated  = ", filename)


if __name__ == "__main__":
    main()
