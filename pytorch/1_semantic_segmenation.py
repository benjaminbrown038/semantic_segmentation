!pip install gitdb2
!pip install GitPython 
!pip install -U albumentations --no-binary qudida, albumentations

import os 
from operator import itemgetter
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import torchvision.models as model
import torch.nn.functional as F

from tqdm.auto import tqdm
from albumentations import Compose, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks
from trainer.utils import setup_system, path_configs, download_git_folder, get_camvid_dataset_parameters, draw_semantic_segmentation_batch, draw_semantic_segmentation_samples, init_semantic_segmentation_dataset
from trainer.base_metric import BaseMetric
from trainer.configuration import SystemConfig, DatasetConfig, TrainerConfig, OptimizerConfig, DataloaderConfig
from trainer.matplotlib_visualizer import MatplotlibVisualizer
from trainer.tensorboard_visualizer import TensorboardVisualizer

class SemSegDataset():
    def __init__():

    def get_num_classes():

    def get_class_name():

    def __len__():

    def __getitem():


test_dataset 
draw_semantic_segmentation_samples

class ConfusionMatrix():
    def __init__():

    def reset():

    def update_value():

    def get_metric_value():


class IntersectionOverUnion():
    def __init__():

    def reset():

    def update_value():

    def get_metric_value():

metric = 
for sample in tqdm(test_dataset,ascii=False):
    masks = sample["mask"]
    metric.update_value(masks,masks)
values = metric.get_metric_value()
print(value['mean_iou'])
print(value['iou'])

class ResNetEncoder():
    def __init__():

    def get_channels_out():

    def forward():

    @staticmethod
    def _get_block_size():

class LateralConnection():
    def __init__():

    def forward():

class FPNDecoder():
    def __init__():
    def forward():

class SemanticSegmentation():
    def __init__():

    def forward():


      



    
  

