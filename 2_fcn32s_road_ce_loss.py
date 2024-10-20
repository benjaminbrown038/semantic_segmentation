import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
import glob as glob
import albumentations as A
import requests
import zipfile

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPool2D, Conv2DTranspose
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

block_plot = False 

def system_config():

@dataclass(frozen=True)
class DatasetConfig:

@dataclass(frozen=True)
class TrainingConfig:

@dataclass(frozen=True)
class InferenceConfig:

def fcn32s_vgg16():
  return model

model = fcn32s_vgg16()
model.summary()

def download_file():

def unzip():

save_name

if not:

class CustomSegDataLoader():
    def __init__():
    def __len__():
    def transforms():
    def resize():
    def reset_array():
    def __getitem__():

id2color = {}
id2color_display = {}

def rgb_to_onehot():
    return arr

def num_to_rgb():
    return 

def image_overlay():
    return 

def display_image_and_mask():
    plt.show()

def create_dataset():
    return train_ds, valid_ds

train_ds, valid_ds = create_datasets(aug=True)

for i, (images,masks) in enumerate(train_ds):
    if == 3:
        break
    image,mask = images,masks
    display_image_and_mask()

model.compile()

if not os.path.exists():
    os.makedirs(TrainingConfig.CHECKPOINT_DIR)
num_versions
version_dir
os.makedirs
model_checkpoint_callback

history = model.fit()

def plot_results():
    plt.close()

train_loss
train_acc
valid_loss
valid_acc

plot_results()
max_loss
plot_results

trained_model
evaluate
print()

def inference

inference()


  

