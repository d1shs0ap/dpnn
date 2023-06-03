import kdtree
from data import *
from dpnn import *
from evaluation import *
from config import *


if __name__ == '__main__':
    
    ### Generate server and client

    # generate server
    server = generate_server_from_random(DIMENSION, SERVER_SIZE, LOWER_BOUND, UPPER_BOUND)
    server_tree = kdtree.create(server, DIMENSION)

    # generate client
    client = generate_client_from_random(DIMENSION, LOWER_BOUND, UPPER_BOUND)



    ### Compute nearest neighbours using different methods

    # retrieve the true nearest neighbours
    true_nn_and_dists = server_tree.search_knn(client, k) # if distance is not given, Euclidean distance is assumed
    true_nn = list(map(lambda node_and_dist: node_and_dist[0], true_nn_and_dists))

    # use LDP to retrieve top NN
    noised_client = list(map(lambda x, y: x + y, client, generate_laplace_noise(DIMENSION, LDP_EPS, sensitivity=UPPER_BOUND - LOWER_BOUND)))
    ldp_nn_and_dists = server_tree.search_knn(noised_client, LDP_K)
    ldp_nn = list(map(lambda node_and_dist: node_and_dist[0], ldp_nn_and_dists))

    # use exp. mechanism to retrieve top NN
    dptt_nn = search_dptt_cmp(server_tree, client, DPTT_NODE_EPS, splits_left=EARLY_STOPPING_LEVEL)



    ### Evaluation

    # precision: percentage of retrieved neighbours that are true neighbours
    ldp_precision = calculate_precision(ldp_nn, true_nn)
    dptt_precision = calculate_precision(dptt_nn, true_nn)
    print(f"LDP precision: {ldp_precision}, DPTT precision: {dptt_precision}")

    # recall: percentage of true neighbours that are retrieved
    ldp_recall = calculate_recall(ldp_nn, true_nn)
    dptt_recall = calculate_recall(dptt_nn, true_nn)
    print(f"LDP recall: {ldp_recall}, DPTT recall: {dptt_recall}")
