import kdtree_extension
import kdtree
import pandas as pd
import inspect
from tqdm import tqdm
import os
import pickle

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
        'server_trial',
        'client_trial',
        'early_stopping_level', 'scheduler_type', # DP-TT
        'eps', # total eps
        'geo_eps', # eps / sensitivity
        'method',
        'return_size',

         # --- evaluation metrics ---
        'raw_acc', 'top_5_acc', 'precision', 'recall',
    ])

    metadata = {
        'dimension': DIMENSION,
        'server_size': SERVER_SIZE,
        'server_domain': SERVER_DOMAIN,
        'client_batch_size': CLIENT_BATCH_SIZE,
        'client_sensitivity': CLIENT_SENSITIVITY,
        'true_radius': int(math.sqrt(TRUE_RADIUS_SQUARED)),
        'early_stopping_levels': EARLY_STOPPING_LEVELS,
        'number_of_nn_within_radius': [],
    }

    # -----------------------------------------------------------------------------
    # ------------------------------ RUN EXPERIMENTS ------------------------------
    # -----------------------------------------------------------------------------
    
    print('Running experiments...')

    # progress bar
    with tqdm(total=SERVER_BATCH_SIZE * CLIENT_BATCH_SIZE * len(EARLY_STOPPING_LEVELS) * len(SCHEDULER_TYPES) * len(EPSILONS)) as pbar:

        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE SERVER ------------------------------
        # -----------------------------------------------------------------------------
        
        for server_trial in range(SERVER_BATCH_SIZE):

            server = generate_server_from_random(DIMENSION, SERVER_SIZE, SERVER_DOMAIN)
            server_tree = kdtree.create(server, DIMENSION) # server tree with no change
            dptt_server_tree = kdtree_extension.create(server, DIMENSION) # server tree with nodes pushed to the leaf


            # -----------------------------------------------------------------------------
            # ------------------------------ GENERATE CLIENT ------------------------------
            # -----------------------------------------------------------------------------

            for client_trial in range(CLIENT_BATCH_SIZE):

                client = generate_client_from_random(DIMENSION, CLIENT_SENSITIVITY)


                # --------------------------------------------------------------------------------------
                # ------------------------------ SEARCH NEAREST NEIGHBOUR ------------------------------
                # --------------------------------------------------------------------------------------

                # True NN search
                top_1_nn = search_true(client, server_tree, 1)
                top_5_nn = search_true(client, server_tree, 5)
                
                nn_within_radius = search_within_radius(client, server_tree, TRUE_RADIUS_SQUARED)
                metadata['number_of_nn_within_radius'].append(len(nn_within_radius))

                for early_stopping_level in EARLY_STOPPING_LEVELS:
                    for scheduler_type in SCHEDULER_TYPE_TO_SCHEDULERS:
                        for scheduler in SCHEDULER_TYPE_TO_SCHEDULERS[scheduler_type]:

                            # DP-TT-CMP search
                            cmp_nn, cmp_eps_lst, cmp_eps_geo_lst  = search_dptt(client=client, server_tree=dptt_server_tree, dimension=DIMENSION, early_stopping_level=early_stopping_level, scheduler=scheduler, cmp=True)
                            eps_cmp = calculate_adaptive_eps(eps_lst=cmp_eps_lst, delta=1/SERVER_SIZE)
                            eps_geo_cmp = sum(cmp_eps_geo_lst)

                            # L-SRR search
                            lsrr_nn = search_lsrr(client=client, server_tree=server_tree, dimension=DIMENSION, eps=eps_cmp, k=len(cmp_nn), domain=SERVER_DOMAIN)

                            # Laplace search, set the k-val (return size) same as return size of DP-TT search
                            laplace_nn = search_laplace(client=client, server_tree=server_tree, dimension=DIMENSION, geo_eps=eps_cmp / CLIENT_SENSITIVITY, k=len(cmp_nn))

                            # # Laplace search with equivalent geo-indistinguishibility epsilon
                            # laplace_geo_nn = search_laplace(client=client, server_tree=server_tree, dimension=DIMENSION, geo_eps=eps_geo_cmp, k=len(cmp_nn))



                            # -----------------------------------------------------------------------------
                            # ------------------------------ EVALUATION -----------------------------------
                            # -----------------------------------------------------------------------------
                            
                            results = pd.DataFrame([
                                # --- DP-TT-CMP ---
                                {
                                    # --- experiment settings ---
                                    'server_trial': server_trial, 'client_trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
                                    'geo_eps': eps_geo_cmp, 'eps': eps_cmp, 'method': 'DP-TT-CMP', 'return_size': len(cmp_nn),

                                    # --- evaluation metrics ---
                                    'raw_acc': calculate_top_k_accuracy(retrieved=cmp_nn, top_k_relevant=top_1_nn), 
                                    'top_5_acc': calculate_top_k_accuracy(retrieved=cmp_nn, top_k_relevant=top_5_nn), 
                                    'precision': calculate_precision(retrieved=cmp_nn, relevant=nn_within_radius), 
                                    'recall': calculate_recall(retrieved=cmp_nn, relevant=nn_within_radius),
                                },
                                # --- L-SRR ---
                                {
                                    # --- experiment settings ---
                                    'server_trial': server_trial, 'client_trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type,
                                    'geo_eps': None, 'eps': eps_cmp, 'method': 'L-SRR', 'return_size': len(cmp_nn),


                                    # --- evaluation metrics ---
                                    'raw_acc': calculate_top_k_accuracy(retrieved=lsrr_nn, top_k_relevant=top_1_nn), 
                                    'top_5_acc': calculate_top_k_accuracy(retrieved=lsrr_nn, top_k_relevant=top_5_nn), 
                                    'precision': calculate_precision(retrieved=lsrr_nn, relevant=nn_within_radius), 
                                    'recall': calculate_recall(retrieved=lsrr_nn, relevant=nn_within_radius),
                                },
                                # --- LAPLACE ---
                                {
                                    # --- experiment settings ---
                                    'server_trial': server_trial, 'client_trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
                                    'geo_eps': eps_cmp / CLIENT_SENSITIVITY, 'eps': eps_cmp, 'method': 'LAPLACE', 'return_size': len(cmp_nn),


                                    # --- evaluation metrics ---
                                    'raw_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_1_nn), 
                                    'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_5_nn), 
                                    'precision': calculate_precision(retrieved=laplace_nn, relevant=nn_within_radius), 
                                    'recall': calculate_recall(retrieved=laplace_nn, relevant=nn_within_radius),
                                },
                                # # --- LAPLACE GEO ---
                                # {
                                #     # --- experiment settings ---
                                #     'server_trial': server_trial, 'client_trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
                                #     'geo_eps': eps_geo_cmp, 'eps': eps_geo_cmp * CLIENT_SENSITIVITY, 'method': 'LAPLACE-GEO', 'return_size': len(cmp_nn),

                                #     # --- evaluation metrics ---
                                #     'raw_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_1_nn), 
                                #     'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_5_nn), 
                                #     'precision': calculate_precision(retrieved=laplace_geo_nn, relevant=nn_within_radius), 
                                #     'recall': calculate_recall(retrieved=laplace_geo_nn, relevant=nn_within_radius),
                                # },
                            ])

                            raw_df = pd.concat([raw_df, results], ignore_index=True)

                            # update progress bar
                            pbar.update(1)


    # -----------------------------------------------------------------------------
    # ------------------------------ SAVE FILES -----------------------------------
    # -----------------------------------------------------------------------------

    # make directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # save files
    raw_df.to_csv(f'{OUTPUT_DIR}/raw.csv')
    with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
