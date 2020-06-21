# from DETR main.py with modifications.
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import math
import sys

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
from custom_criterion import MSLELoss

from dataset import MapQueryDataset

def get_args_parser():
    parser = argparse.ArgumentParser('TransforMap', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Map backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")

     # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # dataset parameters
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)


    return parser

def main(args):
    device = torch.device(args.device)
    
    # Seed
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build the models
    backbone_model = BackboneModel(hidden_dim=args.hidden_dim)
    transformer_model = TransformerModel(
        d_model=args.hidden_dim,
        n_head=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        normalize_before=False
    )
    model = TraMapModel(backbone_model, transformer_model)
    print("DEVICE:", device)
    backbone_model.to(device)
    transformer_model.to(device)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Data loader
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_train = MapQueryDataset(transforms=transforms, split='train')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    num_workers=args.num_workers)

    
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    if args.eval:
        test_stats = None
    
    # Criterion / Loss function
    # criterion = MSLELoss()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss()
    criterion.to(device)

    # Logger thing
    MB = 1024.0 * 1024.0
    print_every = 10
    

    target = data_loader_train
    print("Start Training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        criterion.train()
        print("EPOCH:", epoch)

        i = 0
        ## Training process ##
        # Move to GPU or CPU
        for sample, query, duration in data_returner(data_loader_train):

            query = query.to(device)
            sample = sample.to(device)
            ## Target duration
            duration = duration.to(device)
            duration = duration.float()
            outputs = model(sample, query)
            outputs = outputs.flatten()
            # RMSE if criterion set to MSE
            # loss = torch.sqrt(criterion(outputs, duration) + 1e-6)
            loss = criterion(outputs, duration)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stop the training process".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            
            optimizer.step()

            if i % print_every == 0:
                # print("Output: {} Target: {}".format(outputs.tolist()[0], duration.tolist()[0]))
                if torch.cuda.is_available():
                    print("Iter: {} Memory: {:d}MB Loss: {}".format(i, math.trunc(torch.cuda.max_memory_allocated() / MB), loss_value))
                else:
                    print("Iter: {} Loss:{}".format(i, loss_value))
            i += 1
        lr_scheduler.step()
        ## Saving or Not saving, there is no in between
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def data_returner(iteratable, print_freq=10):
    for obj in iteratable:
        yield obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransfoMap training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)