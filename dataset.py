from pathlib import Path
import glob
import dateutil
from datetime import datetime
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import numpy as np

class MapQueryDataset(Dataset):
    """
    Map Dataset
    """
    def __init__(self, data_dir='./datasets', split='train', transforms=None):
        
        self.data_dir = data_dir 
        self.image_names, self.trajectory = read_json(data_dir+"/query/"+ split +"/*.geojson")
        self._transforms = transforms
        self.split = split
        

    def __len__(self):
        return len(self.trajectory)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_dir + "/image/" + self.split + "/" + self.image_names[idx]
        try:
            image = Image.open(image_path)
            query = self.trajectory[idx]
            if self._transforms is not None:
                image = self._transforms(image)
                query = torch.from_numpy(query)
            
            # queries = [rain_fall_rate, distance, time_of_day, day_of_week]
            # Return image, queries, duration
            return image, query[1:], query[0] 
        except Exception as e:
            self.trajectory = np.delete(self.trajectory, idx)

def read_json(json_path):
    json_paths = glob.glob(json_path)
    json_data = []
    image_names = []
    for json_file in json_paths:
        with open(json_file, "r") as data_file:
            data_loaded = json.load(data_file)
            data_loaded = data_loaded['features'][0]['properties']

            origin_time = dateutil.parser.parse(data_loaded['origin_time'])
            destination_time = dateutil.parser.parse(data_loaded['destination_time'])
            time_of_day = datetime.strftime(origin_time, "%H%M")
            duration =  destination_time - origin_time
            duration = duration.total_seconds()
            day_of_week = origin_time.weekday()
            image_name = data_loaded['trj_id'] + ".png"
            distance = data_loaded['distance'] / 1000 # meters -> kilometers
            rain_fall_rate = data_loaded['rainfall']
            # data[duration, distance, rain_fall_rate, time_of_day, day_of_week]
            query = [
                duration,
                distance,
                rain_fall_rate,
                int(time_of_day),
                day_of_week
            ]
            json_data.append(query)
            image_names.append(image_name)
    return (np.array(image_names), np.array(json_data))