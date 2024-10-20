import numpy as np
import glob as glob 
import matplotlib.pyplot as plt 
import zipfile
import requests
import albumentations as A
import cv2
import os 

from tensorflow.keras.utils import Sequence
from dataclasses import dataclass

def download_file():


def unzip():

@dataclass(fozen=True)
class DatasetConfig:
    NUM_CLASSES
    IMG_HEIGHT
    IMG_WIDTH
    DATA_TRAIN_IMAGES
    DATA_TRAIN_LABELS
    DATA_VALID_IMAGES
    DATA_VALID_LABELS
    DATA_TEST_IMAGES
    DATA_TEST_LABELS

class CustomSegDataLoader(Sequence):
    def __init__():

    def __len__():

    def transforms():

    def resize():

    def reset_array():

    def __getitem__():


id2color = {}

def rgb_to_onehot(rgb_arr, color_map, num_classes):
    return arr

# convert a single channel mask to an rgb mask
def num_to_rgb():


def image_overlay():
    return np.clip(image)

def display_image_and_mask():
    plt.show()


def create_datasets():
    return train_ds, valid_ds

create_datasets()

for i, (images,masks) in enumerate(valid_ds):


