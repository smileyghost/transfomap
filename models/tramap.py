"""
TraMap model
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

class TraMapModel(nn.Module):
    def __init__(self, backbone, transformer):
        '''
        Initializes the model.
        Parameters:
            backbone: Convolutional Layer
            transformer: Transformer Layer
        '''
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.time_embed = nn.Embedding(2400, hidden_dim)
        self.day_embed = nn.Embedding(7, hidden_dim)
        self.rain_embed = nn.Linear(1, hidden_dim)
        self.distance_embed = nn.Linear(1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, samples, query):
        '''
        samples: map images [batch_size x C x H x W]
        '''
        map_features, pos_embed = self.backbone(samples)
        rain = query[:,0]
        distance = query[:,1]
        time = query[:,2].long()
        day = query[:,3].long()
        rain = self.rain_embed(rain.unsqueeze(1).float())
        distance = self.distance_embed(distance.unsqueeze(1).float())

        query_embed = self.time_embed(time) + self.day_embed(day) + rain + distance

        hs = self.transformer(map_features, query_embed, pos_embed)
        output = self.output_layer(hs)
        return output
        
        
