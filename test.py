import json
import glob
import dateutil.parser
from datetime import datetime
from pathlib import Path

def read_json(json_path):
    json_paths = glob.glob(json_path)
    json_data = []
    for json_file in json_paths:
        with open(json_file, "r") as data_file:
            data_loaded = json.load(data_file)
            data_loaded = data_loaded['features'][0]['properties']
            origin_time = dateutil.parser.parse(data_loaded['origin_time'])
            destination_time = dateutil.parser.parse(data_loaded['destination_time'])
            time_of_day = datetime.strftime(origin_time, "%H%M")
            duration =  destination_time - origin_time
            data_loaded['duration'] = duration.total_seconds()
            data_loaded['time_of_day'] = time_of_day
            # json_data[data_loaded['trj_id']] = data_loaded
            json_data.append(data_loaded)

    return json_data

root_dir = Path("./datasets")
trajectory_path = Path(root_dir / "mapmatched")

trajectory = read_json(str(trajectory_path) + "/*.json")
print(trajectory[0]['trj_id'])
