# from DETR main.py with modifications.
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import math
import sys
import os

from PIL import Image
import requests
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from skimage import io


from models.transformer import TransformerModel
from models.tramap import TraMapModel
from models.backbone import BackboneModel

from azureml.core.model import Model

def init():
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim = 256
    nheads = 8
    layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    # Build the models
    backbone_model = BackboneModel(hidden_dim=hidden_dim)
    transformer_model = TransformerModel(
        d_model=hidden_dim,
        n_head=nheads,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        normalize_before=False
    )
    model = TraMapModel(backbone_model, transformer_model)
    backbone_model.to(device)
    transformer_model.to(device)
    model.to(device)

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'checkpoint.pt')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    model.eval()

def run(input_data):
    raw_data = json.loads(input_data['data'])
    
    processed_data = [raw_data['']]
    input_data = torch.tensor(json.loads(input_data['data']))
    # input_data [lat]

    with torch.no_grad():
        output = model(input_data)
        result = {"eta": output}
        return result