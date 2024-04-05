import kdtree_extension
import kdtree
import pandas as pd
import numpy as np
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

    config = density_config_5
    # config = one_dim_config_5
    # config = gowalla_sf_config
    

    raw_df = pd.DataFrame(columns=[
        # --- experiment settings ---
        'trial',
        'early_stopping_level', 'scheduler_type', # DP-TT
        'eps_cmp', # total eps
        'base_eps',
        'geo_eps', # eps / sensitivity
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


        # -----------------------------------------------------------------------------
        # ------------------------------ GENERATE CLIENT ------------------------------
        # -----------------------------------------------------------------------------

        for client_trial in range(config.client_batch_size):

            client = generate_point_from_random(domain=config.domain)

            # --------------------------------------------------------------------------------------
            # ------------------------------ SEARCH NEAREST NEIGHBOUR ------------------------------
            # --------------------------------------------------------------------------------------

            for early_stopping_level, early_stopping_constant in config.early_stopping_levels.items():
                for scheduler_type in config.scheduler_types:
                    for eps in config.epsilons:

                        # DP-TT-CMP
                        cmp_nn, cmp_eps_lst, cmp_eps_geo_lst = search_dptt(client=client, server_tree=dptt_server_tree, early_stopping_level=early_stopping_level, early_stopping_constant=early_stopping_constant, sparsity_constant = config.sparsity_constant, scheduler=scheduler_type(eps))
                        eps_cmp = calculate_adaptive_eps(eps_lst=cmp_eps_lst, delta=1/config.server_size)
                        eps_geo_cmp = sum(cmp_eps_geo_lst)
                        cmp_nn = [node.data for node in cmp_nn]

                        # Laplace search vs. DP-TT-CMP
                        laplace_eps = eps_cmp * 0.5
                        laplace_noised_client = sample_laplace(client=client, geo_eps=laplace_eps / config.client_sensitivity)

                        # L-SRR search and Square Mechanism vs. DP-TT-CMP if dimension is 2
                        if len(config.domain) == 2:
                            lsrr_eps = eps_cmp * 3
                            lsrr_noised_client = sample_lsrr(client=client, server_tree=server_tree, eps=lsrr_eps, domain=config.domain)

                            sm_eps = eps_cmp * 0.1
                            sm_noised_client = sample_sm(client=client, eps=sm_eps, domain=config.domain)
                        
                        # -----------------------------------------------------------------------------
                        # ------------------------------ EVALUATION -----------------------------------
                        # -----------------------------------------------------------------------------
                        
                        results = [
                            # --- DP-TT-CMP ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__, 
                                'eps': eps_cmp, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'DP-TT-CMP', 'return_size': len(cmp_nn),
                                
                                # --- evaluation metrics ---
                                'mse': np.linalg.norm(np.array(client) - np.mean(np.array(cmp_nn), axis=0)),
                            },
                            # --- LAPLACE ---
                            {
                                # --- experiment settings ---
                                'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__, 
                                'eps': laplace_eps, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'LAPLACE', 'return_size': len(cmp_nn),

                                # --- evaluation metrics ---
                                'mse': np.linalg.norm(np.array(client) - np.array(laplace_noised_client)),
                            },
                        ]

                        if len(config.domain) == 2:
                            results.extend([
                                # --- L-SRR ---
                                {
                                    # --- experiment settings ---
                                    'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__,
                                    'eps': lsrr_eps, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'L-SRR', 'return_size': len(cmp_nn),


                                    # --- evaluation metrics ---
                                    'mse': np.linalg.norm(np.array(client) - np.array(lsrr_noised_client)),
                                },
                                # --- Square Mechanism ---
                                {
                                    # --- experiment settings ---
                                    'trial': client_trial, 'early_stopping_level': early_stopping_level, 'scheduler_type': scheduler_type.__name__,
                                    'eps': sm_eps, 'eps_cmp': eps_cmp, 'base_eps': eps, 'method': 'SM', 'return_size': len(cmp_nn),

                                    # --- evaluation metrics ---
                                    'mse': np.linalg.norm(np.array(client) - np.array(sm_noised_client)),
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
