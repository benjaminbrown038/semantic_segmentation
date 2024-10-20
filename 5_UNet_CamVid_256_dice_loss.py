import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
import cv2
import glob as glob
import albumentations as A 
import requests
import zipfile

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Dropout, concatenate, Activation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass


from tensorflow.python.ops.numpy_ops import np_config

def system_config():

@dataclass
class DatasetConfig

@dataclass 
class TrainingConfig

@dataclass 
class InferenceConfig

def unet():
    return model

model = unet()
model.summary()

def download_file():

def unzip()

save_name


class CustomSegDataLoader():
    def __init__():
    def __len__():
    def transforms():
    def resize():
    def reset_array():
    def __getitem__():

id2color = {}

def rgb_to_onehot():

def num_to_rgb():

def image_overlay():

def display_image_and_mask():

def create_datasets():

def mean_iou():
    return mean_iou

def dice_coef_loss():
    return 

model.compile()

if not os.path.exists()
    os.makedirs()

num_versions
version_dir
os.makedirs

model_checkpoint_callback 
history = model.fit()

def plot_results():

train_acc
valid_acc

train_iou
valid_iou

plot_results
plot_results



