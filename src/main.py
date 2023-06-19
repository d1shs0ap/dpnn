import kdtree_extension
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
        'early_stopping_level', 'node_eps_generator', # DP-TT
        'eps', # total eps


         # --- evaluation metrics ---
        'raw_acc_dptt', 'top_5_acc_dptt', 'precision_dptt', 'recall_dptt',
        'raw_acc_ldp', 'top_5_acc_ldp', 'precision_ldp', 'recall_ldp'
    ])

    raw_df.dimension = DIMENSION
    raw_df.server_size = SERVER_SIZE
    raw_df.server_domain = SERVER_DOMAIN
    raw_df.client_batch_size = CLIENT_BATCH_SIZE
    raw_df.client_sensitivity = CLIENT_SENSITIVITY
    raw_df.true_k = TRUE_K
    raw_df.early_stopping_levels = EARLY_STOPPING_LEVELS
    raw_df.node_eps_generators = [inspect.getsource(generator) for generator in NODE_EPS_GENERATORS]


    # -----------------------------------------------------------------------------
    # ------------------------------ RUN EXPERIMENTS ------------------------------
    # -----------------------------------------------------------------------------
    
    print('Running experiments...')

    # progress bar
    with tqdm(total=CLIENT_BATCH_SIZE * len(EARLY_STOPPING_LEVELS) * len(NODE_EPS_GENERATORS)) as pbar:

        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE SERVER ------------------------------
        # -----------------------------------------------------------------------------
        
        server = generate_server_from_random(DIMENSION, SERVER_SIZE, SERVER_DOMAIN)
        server_tree = kdtree_extension.create(server, DIMENSION)
        

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
                for node_eps_generator in NODE_EPS_GENERATORS:
                    
                    # DP-TT search
                    dptt_nn, eps_lst = search_dptt_cmp(client=client, server_tree=server_tree, dimension=DIMENSION, early_stopping_level=early_stopping_level, node_eps_generator=node_eps_generator)
                    eps = calculate_adaptive_eps(eps_lst=eps_lst, delta=1/SERVER_SIZE)

                    # LDP search, set the k-val (return size) same as return size of DP-TT search
                    ldp_nn = search_ldp(client=client, server_tree=server_tree, dimension=DIMENSION, eps=eps, sensitivity=CLIENT_SENSITIVITY, k=len(dptt_nn))

                    # -----------------------------------------------------------------------------
                    # ------------------------------ EVALUATION -----------------------------------
                    # -----------------------------------------------------------------------------
                    
                    # DP-TT results
                    results = pd.DataFrame([{
                        # --- experiment settings ---
                        'dimension': DIMENSION,
                        'server_size': SERVER_SIZE, 'server_domain': SERVER_DOMAIN, # server generation
                        'client_sensitivity': CLIENT_SENSITIVITY, 'trial':trial, # client generation
                        'true_k': TRUE_K, # true NN
                        'early_stopping_level': early_stopping_level, 'node_eps_generator': inspect.getsource(node_eps_generator), # DP-TT
                        'eps': eps,


                        # --- evaluation metrics ---
                        'raw_acc_dptt': calculate_top_k_accuracy(retrieved=dptt_nn, relevant=true_nn, k=1), 
                        'top_5_acc_dptt': calculate_top_k_accuracy(retrieved=dptt_nn, relevant=true_nn, k=5), 
                        'precision_dptt': calculate_precision(retrieved=dptt_nn, relevant=true_nn), 
                        'recall_dptt': calculate_recall(retrieved=dptt_nn, relevant=true_nn),

                        'raw_acc_ldp': calculate_top_k_accuracy(retrieved=ldp_nn, relevant=true_nn, k=1), 
                        'top_5_acc_ldp': calculate_top_k_accuracy(retrieved=ldp_nn, relevant=true_nn, k=5), 
                        'precision_ldp': calculate_precision(retrieved=ldp_nn, relevant=true_nn), 
                        'recall_ldp': calculate_recall(retrieved=ldp_nn, relevant=true_nn)
                    }])

                    raw_df = pd.concat([raw_df, results], ignore_index=True)

                    # update progress bar
                    pbar.update(1)
    
    raw_df.to_csv('graphs/quadratic/raw.csv')
