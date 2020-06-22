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
import dateutil
from pytz import timezone

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
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'checkpoint.pth')
    checkpoint = torch.load(model_path, map_location='cpu')

    args = checkpoint['args']
    hidden_dim = args.hidden_dim
    nheads = args.nheads
    enc_layers = args.enc_layers
    dec_layers = args.dec_layers
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    # Build the models
    backbone_model = BackboneModel(hidden_dim=hidden_dim, arch=args.backbone)
    transformer_model = TransformerModel(
        d_model=hidden_dim,
        n_head=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        normalize_before=False
    )
    model = TraMapModel(backbone_model, transformer_model)
    backbone_model.to(device)
    transformer_model.to(device)
    model.to(device)

    model.load_state_dict(checkpoint['model'])

    model.eval()

def run(input_data):
    raw_data = json.loads(input_data)
    timestamp = datetime.datetime.utcfromtimestamp(raw_data['timestamp'])
    timestamp = timestamp.replace(tzinfo=timezone('Asia/Singapore'))
    timestamp = timestamp.strftime('%Y-%m-%dT%H:%M:%S.000000')
    
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
    
    image_data = Image.open(image_bytes).convert('RGB')
    transforms_image = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Time
    origin_time = dateutil.parser.parse(timestamp)
    time_of_day = datetime.datetime.strftime(origin_time, "%H%M")
    day_of_week = origin_time.weekday()
    # Distance meters to kilometers
    distance = distance / 1000
    # Transform image
    image_data = transforms_image(image_data)
    # Create queries
    query = torch.tensor([
        rainfall,
        distance,
        int(time_of_day),
        int(day_of_week)
    ])
    query = query.unsqueeze(0)
    sample = image_data.unsqueeze(0)

    with torch.no_grad():
        output = model(sample, query)
        result = {"eta": int(output.item())}
        return result
