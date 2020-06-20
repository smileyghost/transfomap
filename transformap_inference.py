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
import io

# Model
from models.transformer import TransformerModel
from models.tramap import TraMapModel
from models.backbone import BackboneModel

# Azure
from azureml.core.model import Model

# Rainfall
from extract_rainfall import getRainfall_coordinates

# Map Generator
from map_generator import map_image_generator

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
    timestamp = datetime.datetime.utcfromtimestamp(raw_data['timestamp']).strftime('%Y-%m-%dT%H:%M:%S.000000')
    
    # rainfall rate in milimeters
    rainfall = getRainfall_coordinates(
        raw_data['latitude_origin'],
        raw_data['longitude_origin'],
        timestamp
    )

    # Return: image in bytes form and distance in kilometers
    image_bytes, distance = map_image_generator(
        sub_key='53EIeobb-HDLQ5KJrW5P6KeeDoKXZFAUlArGW4bwzZc',
        latitude_origin=raw_data['latitude_origin'],
        longitude_origin=raw_data['longitude_origin'],
        latitude_dest=raw_data['latitude_destination'],
        longitude_dest=raw_data['longitude_destination']
    )
    
    image_data = Image.open(image_bytes)
    transforms_image = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(image_data.size)
    image_data = transforms_image(image_data)
    sample = image_data.unsqueeze(1)

    with torch.no_grad():
        output = model(input_data)
        result = {"eta": output}
        return result

inputs = {}
inputs['data'] = '{"latitude_origin": -6.141255, "longitude_origin": 106.692710, "latitude_destination": -6.141150, "longitude_destination": 106.693154, "timestamp": 1590487113}'
run(inputs)