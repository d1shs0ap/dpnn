import kdtree_extension
import kdtree
import pandas as pd
import inspect
from tqdm import tqdm
import os
import pickle

from collections import defaultdict
from data import *
from nn import *
from evaluation import *
from config import *
from budget import *


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------
    # ------------------------ INITIALIZE DATAFRAME AND CONFIG -------------------------
    # ----------------------------------------------------------------------------------

    config = density_config_6
    
    raw_df = pd.DataFrame(columns=[
        # --- experiment settings ---
        'trial',
        'early_stopping_level', 'scheduler_type', # DP-TT
        'eps', # total eps
        'geo_eps', # eps / sensitivity
        'method',
        'return_size',

         # --- evaluation metrics ---
        'raw_acc', 'top_5_acc', 'precision', 'recall',
    ])

    metadata = {
        'dimension': len(config.domain),
        'server_size': config.server_size,
        'config.domain': config.domain,
        'config.client_batch_size': config.client_batch_size,
        'config.client_sensitivity': config.client_sensitivity,
        'true_radius': int(math.sqrt(config.true_radius_squared)),
        'config.early_stopping_levels': config.early_stopping_levels,
        'number_of_nn_within_radius': [],
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
            metadata['number_of_nn_within_radius'].append(len(nn_within_radius))

            for early_stopping_level, early_stopping_constant in config.early_stopping_levels.items():
                for scheduler_type in config.scheduler_type_to_schedulers:
                    for scheduler in config.scheduler_type_to_schedulers[scheduler_type]:

                        # DP-TT-CMP and DIS
                        cmp_nn, cmp_eps_lst, cmp_eps_geo_lst = search_dptt(client=client, server_tree=dptt_server_tree, early_stopping_level=early_stopping_level, early_stopping_constant=early_stopping_constant, scheduler=scheduler)
                        eps_cmp = calculate_adaptive_eps(eps_lst=cmp_eps_lst, delta=1/config.server_size)
                        eps_geo_cmp = sum(cmp_eps_geo_lst)

                        # L-SRR search vs. DP-TT-CMP
                        lsrr_nn = search_lsrr(client=client, server_tree=server_tree, eps=eps_cmp, k=len(cmp_nn), domain=config.domain)

                        # Laplace search vs. DP-TT-CMP
                        laplace_nn = search_laplace(client=client, server_tree=server_tree, geo_eps=eps_cmp / config.client_sensitivity, k=len(cmp_nn))

                        # Laplace search vs. DP-TT-DIS
                        laplace_geo_nn = search_laplace(client=client, server_tree=server_tree, geo_eps=eps_geo_cmp, k=len(cmp_nn))


                        # -----------------------------------------------------------------------------
                        # ------------------------------ EVALUATION -----------------------------------
                        # -----------------------------------------------------------------------------
                        
                        results = pd.DataFrame([
                            # --- DP-TT-CMP ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
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
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type,
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
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
                                'geo_eps': eps_cmp / config.client_sensitivity, 'eps': eps_cmp, 'method': 'LAPLACE', 'return_size': len(cmp_nn),


                                # --- evaluation metrics ---
                                'raw_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_1_nn), 
                                'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_nn, top_k_relevant=top_5_nn), 
                                'precision': calculate_precision(retrieved=laplace_nn, relevant=nn_within_radius), 
                                'recall': calculate_recall(retrieved=laplace_nn, relevant=nn_within_radius),
                            },
                            # --- LAPLACE GEO ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type, 
                                'geo_eps': eps_geo_cmp, 'eps': eps_geo_cmp * config.client_sensitivity, 'method': 'LAPLACE-GEO', 'return_size': len(cmp_nn),

                                # --- evaluation metrics ---
                                'raw_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_1_nn), 
                                'top_5_acc': calculate_top_k_accuracy(retrieved=laplace_geo_nn, top_k_relevant=top_5_nn), 
                                'precision': calculate_precision(retrieved=laplace_geo_nn, relevant=nn_within_radius), 
                                'recall': calculate_recall(retrieved=laplace_geo_nn, relevant=nn_within_radius),
                            },
                        ])

                        raw_df = pd.concat([raw_df, results], ignore_index=True)

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
