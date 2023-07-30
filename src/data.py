import random
import pandas as pd
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


def load_server_from_gowalla_dataset(domain):
    '''
    Load dataset from https://snap.stanford.edu/data/loc-gowalla.html
    '''
    # load data
    df = pd.read_csv('data/loc-gowalla_totalCheckins.txt', sep='\t', header=None)
    df.columns = ['userid','timestamp','latitude','longitude','spotid']

    # select only latitude and longitude
    df = df[['longitude', 'latitude']]

    # choose only points between the latitude and longitude range
    lon_min, lon_max = domain[0]
    lat_min, lat_max = domain[1]
    df = df[(df['latitude'] > lat_min) & (df['latitude'] < lat_max) & (df['longitude'] > lon_min) & (df['longitude'] < lon_max)]

    # convert df to list
    server = list(df.itertuples(index=False, name=None))

    return server

def create_server(config):
    if config.dataset == Dataset.RANDOM:
        server = generate_server_from_random(server_size=config.server_size, domain=config.domain)

    elif config.dataset == Dataset.GOWALLA:
        server = load_server_from_gowalla_dataset(domain=config.domain)
    
    return server