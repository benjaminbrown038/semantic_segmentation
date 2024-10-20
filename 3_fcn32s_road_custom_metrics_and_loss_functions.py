import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import cv2
import glob as glob
import albumentations as A
import requests
import zipfile

from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPool2D, Conv2DTranspose

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

def system_config():

@dataclass(frozen=True)
class DatasetConfig:

@dataclass(frozen=True)
class TrainingConfig:

@dataclass(frozen=True)
class InferenceConfig:

def fcn32s_vgg16():
    return model 

def mean_pixel_accuracy():
    return mean_acc

def mean_iou_naive():
    return mean_iou

def mean_iou():
    return mean_iou

def dice_coef_loss():
    return 1.0 - dc_mean + CCE

model.compile()

if not os.path.exists():
    os.makedirs()
num_versions
version_dir
os.makedirs
model_checkpoint_callback

history = model.fit()

def plot_results():
    plt.close()

train_acc
valid_acc
train_pxa
valid_pxa
train_iou
valid_iou

plot_results()
plot_results()
plot_results()

train_loss
valid_loss
max_loss
plot_results

trained_model
evaluate

print
print
print

def inference():

inference()



