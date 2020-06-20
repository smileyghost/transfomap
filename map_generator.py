# From https://github.com/fnever520/Grab-AzureHackthon/blob/map_render/map_generator.py
import requests
import json
from os import path
import numpy as np
import time
import io

def map_image_generator(sub_key, trj_id=None, latitude_origin=None, longitude_origin=None, latitude_dest=None, longitude_dest=None):
    '''
        Parameters: 
            sub_key : subscription key example => DHFtDtL2CTnX1RKS2bHCPSYJsNm3PfBmmEAMpSFKRwQ
            trj_id : integer value usage: name of the map image example 100
            latitude_origin, longitude_origin, latitude_dest, longitude_dest : float values for 
            latitude and longitude for origin and destination respectively.
        return
            map image trj_id.png as the image extension example 100.png
            routeLengthInMeters float example 12342.5
    '''
    formated_coords = []
    origin_dest = str(latitude_origin)+','+str(longitude_origin)+':'+str(latitude_dest)+','+str(longitude_dest)

    response_data = requests.get('https://atlas.microsoft.com/route/directions/json?subscription-key='+sub_key+'&RouteType=fastest&traffic=true&api-version=1.0&query='+origin_dest)
    routeLengthInMeters = response_data.json()['routes'][0]['summary']['lengthInMeters']

    coordinates = response_data.json()['routes'][0]['legs'][0]['points']
    for cord_pair in coordinates:
        formated_coords.append([cord_pair['longitude'],cord_pair['latitude']])

    data = {
    'type': 'FeatureCollection',
    'features': [
        {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': formated_coords
        },
        'properties': {
            'country': 'Singapore',
            'start': '2019-04-20T04:31:17.000000000',
            'end': '2019-04-20T04:52:28.000000000'
        }
        }
    ]
    }

    max_lng = np.array(data['features'][0]['geometry']['coordinates'])[:,0].max()
    min_lng = np.array(data['features'][0]['geometry']['coordinates'])[:,0].min()
    max_lat = np.array(data['features'][0]['geometry']['coordinates'])[:,1].max()
    min_lat = np.array(data['features'][0]['geometry']['coordinates'])[:,1].min()
    lat = min_lat + (max_lat - min_lat)/2
    lng = min_lng + (max_lng - min_lng)/2

    response = requests.post('https://atlas.microsoft.com/mapData/upload?subscription-key='+sub_key+'&api-version=1.0&dataFormat=geojson', json=data, timeout=30.0)

    if not response.status_code == requests.codes.ok:
        Location = response.headers['Location']
        resp_url = 'https://atlas.microsoft.com/mapData/operations/'
        length = len(resp_url) 
        status_uri = Location[length:-16]

        response_2 = requests.get(resp_url+status_uri+'?api-version=1.0&subscription-key='+sub_key, timeout=30.0) 
        if 'status' in response_2.json():
            while response_2.json()['status'] == 'Running':
                response_2 = requests.get(resp_url+status_uri+'?api-version=1.0&subscription-key='+sub_key, timeout=30.0) 
                
        
        if not response_2.status_code == requests.codes.ok:
            lc = response_2.headers['location'] if response_2.headers['location'] else response_2.headers['Location']
            ln = len('https://atlas.microsoft.com/mapData/metadata/')
            
            udId = lc[ln:-16]

            style = 'layer=basic&style=dark&zoom=11&center='+str(lng)+'%2C'+str(lat)+'&path=lcf78731|fc0000FF|lw4|la1.0|fa1.0||'
            map_response = requests.get('https://atlas.microsoft.com/map/static/png?subscription-key='+sub_key+'&api-version=1.0&'+style+'udid-'+udId, timeout=30.0)

            requests.delete('https://atlas.microsoft.com/mapData/'+udId+'?api-version=1.0&subscription-key='+sub_key)
            
            return io.BytesIO(map_response.content), routeLengthInMeters

        else:
            return 'failed', routeLengthInMeters
    else:
        return 'failed', routeLengthInMeters

# Example on how to call the function 
# map_image_name, distance = map_image_generator('53EIeobb-HDLQ5KJrW5P6KeeDoKXZFAUlArGW4bwzZc', 1000, latitude_origin='1.370959', longitude_origin='103.872222', latitude_dest='1.373563', longitude_dest='103.772978')
# print(map_image_name, distance)