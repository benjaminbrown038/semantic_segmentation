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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, INput, Conv2DTranspose, Activation, Dropout, concatenate
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

def unet():
    return model

model = unet()
model.summary()

def download_file():

def unzip():

save_name

if not

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

def num_to_rgb():

def image_overlay():

def display_image_and_mask():

def create_datasets():

for i, (images,masks) in enumerate(train_ds):
      if i == 3:
          break
      image,mask = images[0], masks[0]
      display_image_and_mask()

def mean_iou():
    return mean_iou

model.compile()
if not os.path.exists()
    os.makedirs()
num_versions
version_dir
os.makedirs()
model_checkpoint_callback

history = model.fit()

def plot_results():

train_acc
valid_acc

train_iou
valid_iou

plot_results
plot_results

train_loss
valid_loss

max_loss
plot_results

trained_model = 
evaluate 
print
print

def inference():

inference()

    
