# generate random locations / read locations from dataset
import random

def generate_server_from_gowalla_dataset():
    pass

def generate_server_from_random(dimension, server_size, lower_bound, upper_bound):
    '''
    :dimension: dimension of each point in the server (e.g., the dimension of a point on the map is 2)
    :size: size of the server to be generated
    :lower_bound: the same lower bound is used for each dimension of a point
    :upper_bound: ^
    
    Generate a list of vectors randomly.
    '''
    server = []

    for _ in range(server_size):
        vector = [random.uniform(lower_bound, upper_bound) for _ in range(dimension)]
        server.append(vector)

    return server

def generate_client_from_random(dimension, lower_bound, upper_bound):
    '''
    :dimension: dimension of the client query (e.g., the dimension of a point on the map is 2)
    :lower_bound: the same lower bound is used for each dimension of a point
    :upper_bound: ^
    '''
    return [random.uniform(lower_bound, upper_bound) for _ in range(dimension)]
