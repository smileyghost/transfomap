import torch
import torch.nn.functional as F
from torch import nn
import math

'''
Transformer Model
'''

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        Parameters:
            d_model: model dimensions
            n_head: number of multi-head self-attention
            num_encoder_layers: number of layers in encoder
            num_decoder_layers: number of layers in decoder
            dim_feedforward: FCN layer dimensions
            activation: activation function
        """
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward,
                                                    dropout, activation)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward,
                                                    dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        
        self.__reset__parameters()

    def __reset__parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, query_embed, pos_embed):
        # src [batch_size x C x H x W] to [HW x batch_size x C]

        src = src.flatten(2).permute(2,0,1)
        pos_embed = pos_embed.flatten(2).permute(2,0,1)
        src = src + pos_embed
        query_embed = query_embed.unsqueeze(1)
        query_embed = query_embed.float()
        # query [ batch_size x 1 x dim_size] to [1 x batch_size x dim_size]
        query_embed = query_embed.permute(1,0,2)
        memory = self.transformer_encoder(src)
        hs = self.transformer_decoder(query_embed, memory)
        return hs
