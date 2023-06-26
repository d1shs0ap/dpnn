import kdtree_extension
import kdtree
import pandas as pd
import inspect
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import *
from nn import *
from evaluation import *
from config import *
from budget import *


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------
    # ------------------------------ INITIALIZE DATAFRAME ------------------------------
    # ----------------------------------------------------------------------------------
    
    raw_df = pd.DataFrame(columns=[
        # --- experiment settings ---
        'trial',
        'early_stopping_level', 'node_geo_eps_generator', # DP-TT
        'geo_eps', # total eps
        'method',

         # --- evaluation metrics ---
        'raw_acc', 'top_5_acc', 'precision', 'recall',
    ])

    raw_df.dimension = DIMENSION
    raw_df.server_size = SERVER_SIZE
    raw_df.server_domain = SERVER_DOMAIN
    raw_df.client_batch_size = CLIENT_BATCH_SIZE
    raw_df.client_sensitivity = CLIENT_SENSITIVITY
    raw_df.true_k = TRUE_K
    raw_df.early_stopping_levels = EARLY_STOPPING_LEVELS
    raw_df.node_geo_eps_generators = [inspect.getsource(generator) for generator in NODE_GEO_EPS_GENERATORS]


    # -----------------------------------------------------------------------------
    # ------------------------------ RUN EXPERIMENTS ------------------------------
    # -----------------------------------------------------------------------------
    
    print('Running experiments...')

    # progress bar
    with tqdm(total=CLIENT_BATCH_SIZE * len(EARLY_STOPPING_LEVELS) * len(NODE_GEO_EPS_GENERATORS)) as pbar:

        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE SERVER ------------------------------
        # -----------------------------------------------------------------------------
        
        server = generate_server_from_random(DIMENSION, SERVER_SIZE, SERVER_DOMAIN)
        server_tree = kdtree.create(server, DIMENSION) # server tree with no change
        dptt_server_tree = kdtree_extension.create(server, DIMENSION) # server tree with nodes pushed to the leaf


        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE CLIENT ------------------------------
        # -----------------------------------------------------------------------------

        for trial in range(CLIENT_BATCH_SIZE):

            # generate client
            client = generate_client_from_random(DIMENSION, CLIENT_SENSITIVITY)


            # --------------------------------------------------------------------------------------
            # ------------------------------ SEARCH NEAREST NEIGHBOUR ------------------------------
            # --------------------------------------------------------------------------------------

            # True NN search
            true_nn = search_true(client, server_tree, TRUE_K)

            for early_stopping_level in EARLY_STOPPING_LEVELS:
                for node_geo_eps_generator in NODE_GEO_EPS_GENERATORS:

                    # DP-TT-CMP search
                    cmp_nn, cmp_geo_eps_lst = search_dptt(client=client, server_tree=dptt_server_tree, dimension=DIMENSION, early_stopping_level=early_stopping_level, node_geo_eps_generator=node_geo_eps_generator, cmp=True, sensitivity=CLIENT_SENSITIVITY)
                    geo_eps_cmp = calculate_adaptive_eps(eps_lst=cmp_geo_eps_lst, delta=1/SERVER_SIZE)
                    

                    # DP-TT-DIS
                    dis_nn, dis_geo_eps_lst = search_dptt(client=client, server_tree=dptt_server_tree, dimension=DIMENSION, early_stopping_level=early_stopping_level, node_geo_eps_generator=node_geo_eps_generator, cmp=False, sensitivity=CLIENT_SENSITIVITY)
                    geo_eps_dis = calculate_eps_sum(dis_geo_eps_lst)
                    

                    # laplace search, set the k-val (return size) same as return size of DP-TT search
                    laplace_nn = search_laplace(client=client, server_tree=server_tree, dimension=DIMENSION, geo_eps=geo_eps_dis, k=len(dis_nn))


                    # -----------------------------------------------------------------------------
                    # ------------------------------ EVALUATION -----------------------------------
                    # -----------------------------------------------------------------------------
                    
                    # DP-TT results
                    results = pd.DataFrame([
                        {
                            # --- experiment settings ---
                            'trial': trial, 'early_stopping_level': early_stopping_level, 'node_geo_eps_generator': inspect.getsource(node_geo_eps_generator), 
                            'geo_eps': geo_eps_cmp, 'method': 'DP-TT-CMP',

                            # --- evaluation metrics ---
                            'raw_acc': calculate_top_k_accuracy(retrieved=cmp_nn, relevant=true_nn, k=1), 
                            'top_5_acc': calculate_top_k_accuracy(retrieved=cmp_nn, relevant=true_nn, k=5), 
                            'precision': calculate_precision(retrieved=cmp_nn, relevant=true_nn), 
                            'recall': calculate_recall(retrieved=cmp_nn, relevant=true_nn),
                        },
                        {
                            # --- experiment settings ---
                            'trial': trial, 'early_stopping_level': early_stopping_level, 'node_geo_eps_generator': inspect.getsource(node_geo_eps_generator), 
                            'geo_eps': geo_eps_dis, 'method': 'DP-TT-DIS',

                            # --- evaluation metrics ---
                            'raw_acc': calculate_top_k_accuracy(retrieved=dis_nn, relevant=true_nn, k=1), 
                            'top_5_acc': calculate_top_k_accuracy(retrieved=dis_nn, relevant=true_nn, k=5), 
                            'precision': calculate_precision(retrieved=dis_nn, relevant=true_nn), 
                            'recall': calculate_recall(retrieved=dis_nn, relevant=true_nn),
                        },
                        {
                            # --- experiment settings ---
                            'trial': trial, 'early_stopping_level': early_stopping_level, 'node_geo_eps_generator': inspect.getsource(node_geo_eps_generator), 
                            'geo_eps': geo_eps_dis, 'method': 'LAPLACE',


                            # --- evaluation metrics ---
                            'raw_acc': calculate_top_k_accuracy(retrieved=laplace_nn, relevant=true_nn, k=1), 
                            'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_nn, relevant=true_nn, k=5), 
                            'precision': calculate_precision(retrieved=laplace_nn, relevant=true_nn), 
                            'recall': calculate_recall(retrieved=laplace_nn, relevant=true_nn),
                        }
                    ])

                    raw_df = pd.concat([raw_df, results], ignore_index=True)

                    # update progress bar
                    pbar.update(1)
    
    raw_df.to_csv('graphs/test/raw.csv')
