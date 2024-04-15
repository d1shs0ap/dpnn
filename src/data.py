import random
import pandas as pd
import numpy as np
from enum import Enum

class Dataset(Enum):
    RANDOM = 0
    GOWALLA = 1

def generate_point_from_random(domain):
    '''
    :domain: (lower, upper)[] for each dimension
    '''
    return [random.uniform(lower, upper) for lower, upper in domain]

def generate_server_from_random(server_size, domain):
    '''
    :server_size: size of the server to be generated
    :domain: (lower, upper)[] for each dimension
    '''
    return [generate_point_from_random(domain) for _ in range(server_size)]

def lat_long_to_km(lat, long):
    long = 111.320 * np.cos(lat * np.pi / 180) * long
    lat *= 110.574
    return lat, long

def load_server_from_gowalla_dataset(domain):
    '''
    Load dataset from https://snap.stanford.edu/data/loc-gowalla.html
    '''
    # load data
    df = pd.read_csv('data/loc-gowalla_totalCheckins.txt', sep='\t', header=None)
    df.columns = ['userid','timestamp','latitude','longitude','spotid']
    print("Number of points:", len(df))

    # select only latitude and longitude, and drop duplicates
    # df = df.drop_duplicates(subset=['spotid'], keep='first')
    df = df[['longitude', 'latitude']]
    df.drop_duplicates(inplace=True)
    
    # choose only points between the latitude and longitude range
    lon_min, lon_max = domain[0]
    lat_min, lat_max = domain[1]
    df = df[(df['latitude'] > lat_min) & (df['latitude'] < lat_max) & (df['longitude'] > lon_min) & (df['longitude'] < lon_max)]

    # transform latitude and longitude into km
    df['latitude'], df['longitude'] = lat_long_to_km(df['latitude'], df['longitude'])
    print("Number of unique points:", len(df))

    # convert df to list
    server = list(df.itertuples(index=False, name=None))

    return server

def create_server(config):
    if config.dataset == Dataset.RANDOM:
        server = generate_server_from_random(server_size=config.server_size, domain=config.domain)

    elif config.dataset == Dataset.GOWALLA:
        server = load_server_from_gowalla_dataset(domain=config.domain)

        # transform config domain
        long0, long1 = config.domain[0]
        lat0, lat1 = config.domain[1]
        lat0, long0 = lat_long_to_km(lat0, long0)
        lat1, long1 = lat_long_to_km(lat1, long1)
        config.domain = [(long0, long1), (lat0, lat1)]
    
    return server