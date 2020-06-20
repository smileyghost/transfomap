# From https://github.com/smileyghost/getRainfall/blob/master/get_rainfall.py
import requests
from datetime import datetime
import pandas as pd
from os import path
from math import radians, degrees, sin, cos, asin, acos, sqrt

def haversine(lat1, lng1, lat2, lng2):
    '''
    calculate the great circle distance for the shortest distance between the coordinate_x with the referenced tower.
    '''
    r = 6371  # in kilometer unit
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    distanceLng = lng2 - lng1
    distanceLat = lat2 - lat1
    a = sin(distanceLat/2)**2 + cos(lat1)*cos(lat2)*sin(distanceLng/2)**2

    return 2*r*asin(sqrt(a))

def nearest_distance(distance_entries, ref):
    return min(distance_entries, key=lambda p: haversine(ref['latitude'], ref['longitude'], p['latitude'], p["longitude"]))

def convert_dt_(time_string):
    datetime_object = datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.000000")
    time = datetime.strftime(datetime_object, "%Y-%m-%dT%H:%M:%S")
    return time

def get_deviceID(getClosestCoordinate, content_station):
    for station in content_station:
        if getClosestCoordinate == station['location']:
            return station['device_id']

def get_rainfall(device_id, content_readings):
    for rd in content_readings:
        if rd['station_id'] == device_id:
            return rd['value']

def getRainfall_coordinates(longitude, latitude, query_time):
    '''
    eg. getRainfall_coordinates(103.8191, 1.3191, "2019-04-20T12:00:30.000000")
    '''
    url = 'https://api.data.gov.sg/v1/environment/rainfall'
    query_time = convert_dt_(query_time)
    payload = {
        "date_time": query_time
    }
    response = requests.get(url, params = payload)
    content = response.json()
    content_station = content['metadata']['stations']
    content_readings = content['items'][0]['readings']

    coordinate = [content_station[i]['location'] for i in range(len(content_station))]

    ref = {
        "latitude": latitude,
        "longitude": longitude
    }

    getClosestCoordinate = nearest_distance(coordinate, ref)

    device_id = get_deviceID(getClosestCoordinate, content_station)
    rainfall = get_rainfall(device_id, content_readings)

    return rainfall