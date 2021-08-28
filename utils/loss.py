import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils.metrics import *
from utils import helpers


class BinarySoftDiceLoss(_Loss):

    def __init__(self):
        super(BinarySoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_dice = diceCoeff(y_pred, y_true)
        return 1 - mean_dice


class SoftDiceLoss(_Loss):

    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice



class BCE_Dice_Loss(_Loss):
    def __init__(self, num_classes, smooth=0, weight=[1.0, 1.0]):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.weight = weight
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        return self.weight[0] * self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes) + self.weight[1] * self.dice_loss(inputs, targets)















