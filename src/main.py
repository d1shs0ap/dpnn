import kdtree_extension
import pandas as pd
from collections import defaultdict
from data import *
from nn import *
from evaluation import *
from config import *
from budget import *


if __name__ == '__main__':
    
    df = pd.DataFrame(columns=[
        'method', 'dimension', 'eps', 'sensitivity', 'server_size', 'client_batch_size', # all method will have these attributes
        'node_eps', 'early_stopping_level', # for tree traversal methods
        'ldp_k', # for LDP
        'raw_acc', 'top_5_acc', 'true_k', 'precision', 'recall' # evaluation metrics
    ])
    eps_to_node_eps = calculate_eps_to_node_eps(NODE_EPS_FUNCTIONS, EARLY_STOPPING_LEVEL)

    # for each generated server and client, do searches w/. different epsilons
    for server_size in SERVER_SIZES:
        for sensitivity in SENSITIVITIES:
            # generate server
            server = generate_server_from_random(DIMENSION, server_size, sensitivity)
            server_tree = kdtree_extension.create(server, DIMENSION)

            eps_to_ldp_results, eps_to_dptt_results = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))

            for _ in range(CLIENT_BATCH_SIZE):

                # generate client
                client = generate_client_from_random(DIMENSION, sensitivity)

                # retrieve the true nearest neighbours
                true_nn = search_true(client, server_tree, TRUE_K)

                for eps, node_eps in eps_to_node_eps.items():

                    # retrieve nearest neighbours
                    ldp_nn = search_ldp(client, server_tree, DIMENSION, eps, sensitivity, LDP_K)
                    dptt_nn = search_dptt_cmp(client, server_tree, DIMENSION, node_eps)
                    
                    # evaluate metrics
                    evaluate(eps_to_ldp_results, eps, ldp_nn, true_nn)
                    evaluate(eps_to_dptt_results, eps, dptt_nn, true_nn)


            for eps, results in eps_to_ldp_results.items():
                results_dict = {
                    'method': 'LDP', 'dimension': DIMENSION, 'eps': eps, 'sensitivity': sensitivity, 'server_size': server_size, 'client_batch_size': CLIENT_BATCH_SIZE,
                    'ldp_k': LDP_K,
                    'raw_acc': calculate_mean(results['raw_acc']), 'top_5_acc': calculate_mean(results['top_5_acc']), 'true_k': TRUE_K, 'precision': calculate_mean(results['precision']), 'recall': calculate_mean(results['recall'])
                }
                df.loc[len(df)] = results_dict

            for eps, results in eps_to_dptt_results.items():
                results_dict = {
                    'method': 'DP-TT', 'dimension': DIMENSION, 'eps': eps, 'sensitivity': sensitivity, 'server_size': server_size, 'client_batch_size': CLIENT_BATCH_SIZE,
                    'node_eps': eps_to_node_eps[eps], 'early_stopping_level': EARLY_STOPPING_LEVEL,
                    'raw_acc': calculate_mean(results['raw_acc']), 'top_5_acc': calculate_mean(results['top_5_acc']), 'true_k': TRUE_K, 'precision': calculate_mean(results['precision']), 'recall': calculate_mean(results['recall'])
                }
                df.loc[len(df)] = results_dict
    
    print(df)
