import kdtree_extension
import kdtree
import pandas as pd
import numpy as np
import inspect
from tqdm import tqdm
import os
import pickle
import time

from collections import defaultdict
from data import *
from nn import *
from evaluation import *
from config import *
from budget import *
from adversary import *


if __name__ == '__main__':


    # ----------------------------------------------------------------------------------
    # ------------------------ INITIALIZE DATAFRAME AND CONFIG -------------------------
    # ----------------------------------------------------------------------------------

    # config = density_config_6
    # config = one_dim_config_5
    config = gowalla_sf_config
    # config = scheduler_config
    

    raw_df = pd.DataFrame(columns=[
        # --- experiment settings ---
        'trial',
        'early_stopping_level', 'scheduler_type', # DP-TT
        'eps_cmp', # total eps
        'base_eps',
        'eps',
        'method',
        'return_size',
        'mse',

         # --- evaluation metrics ---
        'raw_acc', 'top_5_acc', 'precision', 'recall',
    ])

    metadata = {
        'dimension': len(config.domain),
        'server_size': config.server_size,
        'domain': config.domain,
        'client_batch_size': config.client_batch_size,
        'client_sensitivity': config.client_sensitivity,
        'true_radius': int(math.sqrt(config.true_radius_squared)),
        'early_stopping_levels': config.early_stopping_levels,
        'number_of_nn_within_radius': [],
        'epsilons': config.epsilons,
    }

    # -----------------------------------------------------------------------------
    # ------------------------------ RUN EXPERIMENTS ------------------------------
    # -----------------------------------------------------------------------------
    
    print('Running experiments...')

    # progress bar
    with tqdm(total=config.client_batch_size * len(config.early_stopping_levels) * len(config.scheduler_types) * len(config.epsilons)) as pbar:

        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE SERVER ------------------------------
        # -----------------------------------------------------------------------------

        # generate/load points in server
        server = create_server(config)
        
        # create kd-tree from points
        server_tree = kdtree.create(server) # server tree with no change
        dptt_server_tree = kdtree_extension.create(server) # server tree with nodes pushed to the leaf
        # server_rectangles = find_bounding_rectangles(dptt_server_tree, config.domain)


        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE CLIENT ------------------------------
        # -----------------------------------------------------------------------------

        for client_trial in range(config.client_batch_size):

            client = generate_point_from_random(domain=config.domain)


            # --------------------------------------------------------------------------------------
            # ------------------------------ SEARCH NEAREST NEIGHBOUR ------------------------------
            # --------------------------------------------------------------------------------------

            # find top 1 / 5 nearest neighbours of client
            top_1_nn = search_true(client, server_tree, 1)
            top_5_nn = search_true(client, server_tree, 5)
            
            # find points in server within a certain radius of client
            nn_within_radius = search_within_radius(client, server_tree, config.true_radius_squared)
            # if there's nothing around, skip this iteration
            if len(nn_within_radius) == 0:
                pbar.update(len(config.epsilons) * len(config.scheduler_types) * len(config.early_stopping_levels))
                continue
            # update metadata
            metadata['number_of_nn_within_radius'].append(len(nn_within_radius))

            for early_stopping_level, early_stopping_constant in config.early_stopping_levels.items():
                for scheduler_type in config.scheduler_types:
                    for i, eps in enumerate(config.epsilons):

                        # DP-TT-CMP and DIS
                        cmp_nn, cmp_eps_lst, cmp_eps_geo_lst = search_dptt(client=client, server_tree=dptt_server_tree, early_stopping_level=early_stopping_level, early_stopping_constant=early_stopping_constant, sparsity_constant = config.sparsity_constant, scheduler=scheduler_type(eps))
                        eps_cmp = calculate_adaptive_eps(eps_lst=cmp_eps_lst, delta=1/config.server_size)
                        # eps_cmp = sum(cmp_eps_lst)
                        # eps_geo_cmp = sum(cmp_eps_geo_lst)
                        cmp_nn = [node.data for node in cmp_nn]

                        # Laplace search vs. DP-TT-CMP

                        laplace_eps = config.laplace_epsilons[i]
                        laplace_noised_client = sample_laplace(client=client, geo_eps=laplace_eps / config.client_sensitivity)
                        laplace_nn = search_true(client=laplace_noised_client, server_tree=server_tree, k=len(cmp_nn))

                        # # Laplace search vs. DP-TT-DIS
                        # laplace_geo_nn = search_laplace(client=client, server_tree=server_tree, geo_eps=eps_geo_cmp, k=len(cmp_nn))

                        # L-SRR search and Square Mechanism vs. DP-TT-CMP if dimension is 2
                        if len(config.domain) == 2:
                            # lsrr_eps = config.lsrr_epsilons[i]
                            # lsrr_noised_client = sample_lsrr(client=client, server_tree=server_tree, eps=lsrr_eps, domain=config.domain)
                            # lsrr_nn = search_true(client=lsrr_noised_client, server_tree=server_tree, k=len(cmp_nn))

                            sm_eps = config.sm_epsilons[i]
                            sm_noised_client = sample_sm(client=client, eps=sm_eps, domain=config.domain)
                            sm_nn = search_true(client=sm_noised_client, server_tree=server_tree, k=len(cmp_nn))
                        
                        # -----------------------------------------------------------------------------
                        # ------------------------------ EVALUATION -----------------------------------
                        # -----------------------------------------------------------------------------
                        
                        results = [
                            # --- DP-TT-CMP ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__, 
                                'eps': eps_cmp, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'DP-TT-CMP', 'return_size': len(cmp_nn),
                                'mse': np.linalg.norm(np.array(client) - np.mean(np.array(cmp_nn), axis=0)),

                                # --- evaluation metrics ---
                                'raw_acc': calculate_top_k_accuracy(retrieved=cmp_nn, top_k_relevant=top_1_nn), 
                                'top_5_acc': calculate_top_k_accuracy(retrieved=cmp_nn, top_k_relevant=top_5_nn), 
                                'precision': calculate_precision(retrieved=cmp_nn, relevant=nn_within_radius), 
                                'recall': calculate_recall(retrieved=cmp_nn, relevant=nn_within_radius),
                            },
                            # --- LAPLACE ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__, 
                                'eps': laplace_eps, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'LAPLACE', 'return_size': len(cmp_nn),
                                'mse': np.linalg.norm(np.array(client) - np.array(laplace_noised_client)),

                                # --- evaluation metrics ---
                                'raw_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_1_nn), 
                                'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_5_nn), 
                                'precision': calculate_precision(retrieved=laplace_nn, relevant=nn_within_radius), 
                                'recall': calculate_recall(retrieved=laplace_nn, relevant=nn_within_radius),
                            },
                            # # --- LAPLACE GEO ---
                            # {
                            #     # --- experiment settings ---
                            #     'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__, 
                            #     'geo_eps': eps_geo_cmp, 'eps_cmp': eps_geo_cmp * config.client_sensitivity, 'base_eps': eps, 'method': 'LAPLACE-GEO', 'return_size': len(cmp_nn),

                            #     # --- evaluation metrics ---
                            #     'raw_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_1_nn), 
                            #     'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_5_nn), 
                            #     'precision': calculate_precision(retrieved=laplace_geo_nn, relevant=nn_within_radius), 
                            #     'recall': calculate_recall(retrieved=laplace_geo_nn, relevant=nn_within_radius),
                            # },
                        ]

                        if len(config.domain) == 2:
                            results.extend([
                                # # --- L-SRR ---
                                # {
                                #     # --- experiment settings ---
                                #     'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__,
                                #     'eps': lsrr_eps , 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'L-SRR', 'return_size': len(cmp_nn),
                                #     'mse': np.linalg.norm(np.array(client) - np.array(lsrr_noised_client)),


                                #     # --- evaluation metrics ---
                                #     'raw_acc': calculate_top_k_accuracy(retrieved=lsrr_nn, top_k_relevant=top_1_nn), 
                                #     'top_5_acc': calculate_top_k_accuracy(retrieved=lsrr_nn, top_k_relevant=top_5_nn), 
                                #     'precision': calculate_precision(retrieved=lsrr_nn, relevant=nn_within_radius), 
                                #     'recall': calculate_recall(retrieved=lsrr_nn, relevant=nn_within_radius),
                                # },
                                # --- Square Mechanism ---
                                {
                                    # --- experiment settings ---
                                    'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__,
                                    'eps': sm_eps, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'SM', 'return_size': len(cmp_nn),
                                    'mse': np.linalg.norm(np.array(client) - np.array(sm_noised_client)),

                                    # --- evaluation metrics ---
                                    'raw_acc': calculate_top_k_accuracy(retrieved=sm_nn, top_k_relevant=top_1_nn), 
                                    'top_5_acc': calculate_top_k_accuracy(retrieved=sm_nn, top_k_relevant=top_5_nn), 
                                    'precision': calculate_precision(retrieved=sm_nn, relevant=nn_within_radius), 
                                    'recall': calculate_recall(retrieved=sm_nn, relevant=nn_within_radius),
                                },
                            ])
                        
                        results_df =  pd.DataFrame(results)
                        raw_df = pd.concat([raw_df, results_df], ignore_index=True)

                        # update progress bar
                        pbar.update(1)


    # -----------------------------------------------------------------------------
    # ------------------------------ SAVE FILES -----------------------------------
    # -----------------------------------------------------------------------------

    # make directory
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # save files
    raw_df.to_csv(f'{config.output_dir}/raw.csv')
    with open(f'{config.output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Experiment results saved.")
