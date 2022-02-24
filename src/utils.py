#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import cv2
import matplotlib.pyplot as plt
import imutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_reset(model):
    """
    Reset model weights
    args: model object
    returns a model with reset weight parameters
    can be called as: model.apply(weight_reset)
    """
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        model.reset_parameters()


def weight_init(model):
    """
    Initialize model weights (Xavier initialization)
    args: model object
    returns a model with weight parameters bearing Xavier
    initialization
    can be called as: model.apply(weight_init)
    """
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight.data)


def visualize(img_path, gt_label, predicted_label):
    """
    Visualize gaze predictions
    args:       img_path (string) -> file path of road-view image
                gt_label (tuple of ints) -> ground truth gaze coordinates
                predicted_label (tuple of ints)-> predicted gaze coordinates
    """

    # point settings
    radius = 20
    gt_point_color = (0, 255, 0) # green ground truth
    predicted_point_color = (0, 0, 255) # red predicted label
    thickness = -1

    image1 = cv2.imread(img_path)
    image1 = cv2.circle(image1, gt_label, radius, gt_point_color, thickness)
    image = cv2.circle(image1, predicted_label, radius, predicted_point_color, thickness)

    cv2.imshow(image1)


def outputs_and_labels(loader):
    """
    Prints outputs and corresponding labels for all batches
    args:       loader -> dataloader object
    """

    outputs_l = []
    i = 0
    for image_list, labels in loader:
        i += 1
        if (i > 10):
            break
        image_list[0] = image_list[0].to(device=device)
        image_list[1] = image_list[1].to(device=device)
        outputs = outputs[None].float()
        labels = labels.float()
        print('Outputs = ', outputs)
        print('Labels = ', labels)
