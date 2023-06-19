# A Python implementation of DPNN query with exponential mechanism, random response and the early stop mechanism.
import kdtree
import pickle
import sys
import random

import numpy as np
import math

import scipy.special as sk
from datetime import datetime

from prettytable import PrettyTable

import ray
import csv
from decimal import *

class Server:
    '''
    Contains kd tree of locations
    '''
    def __init__(self):
        self.traverse_done = False

    def setup_tree_real_db(self):
        '''
        Create kd tree from a real dataset
        '''
        name_dataset = "gowalla"
        filename_out = 'user_data_%s_discrete.pkl' % name_dataset

        with open(filename_out, 'rb') as f:
            (dict_users,
                translate_loc_id_to_latlon,
                translate_loc_id_to_cartesian,
                latlon_id_grid,
                translate_loc_id_to_latlon_id) = pickle.load(f)

        latlong_user_list = []
        for users in dict_users:
            user = dict_users[users]["traces"][0]
            for location in user:
                latlong_user_list.append(translate_loc_id_to_latlon[int(location)])

        self.tree = kdtree.create(latlong_user_list)

    def tree_tuple(self, domain):
        '''
        Used in the next method to generate a "random" kd tree
        '''
        tree_node_list = []
        for x in range(self.dimension):
            tree_node_list.append(random.randint(0, domain))
        return tuple(tree_node_list)
    
    # Create tree
    def setup_tree(self, domain, db_size, dimension):
        '''
        Create kd tree with random locations
        '''
        self.dimension = dimension
        self.list = [self.tree_tuple(domain) for i in range(db_size)]
        range_lists =[]
        for x in range(dimension):
            self.list.sort(key=lambda a: a[x])
            min_range = self.list[0][x]
            max_range = self.list[-1][x]
            range_lists.append((min_range, max_range))
        client_val_list = []
        # Create random value to be client val
        for ranges in range_lists:
            client_val_list.append(random.randint(ranges[0],ranges[1]))

        self.set_client_value(tuple(client_val_list))
        self.tree = kdtree.create(self.list, dimension)
        
    def set_client_value(self, client_val):
        self.client_val = client_val

    # Compute Euclidian distance
    def compute_distance(self, root_val):
        '''
        compute the euclidean distance between the root node and the client location
        '''
        pre_sqrt_val = 0
        for x in range(self.dimension):
            pre_sqrt_val += (root_val[x] - self.client_val[x])**2
        return math.sqrt(pre_sqrt_val)

    # Compute distacne in one dimension
    def compute_distance_1D(self, root_val, dimension):
        return abs(root_val[dimension]-self.client_val[dimension])

    # Check whehter tree val is smaller than client
    def compute_less(self, root_comp, client_comp):
        if root_comp <= client_comp:
            return 1
        else:
            return 0

    def traverse(self, tree, client_val, eps, sensitivity, splitsLeft=0, search_index = 0):
        '''
        traverse the kdtree to find nearest neighbours of the client location.
        it uses RR_query to add noise to the query result, and will perform early stopping based on given level to stop.
        this method will return a list of NN results and a list of all epsilon cost of this traverse.

        splitsLeft: used to split later for early stopping
        '''
        root = self.tree
        if tree: # so tree can be None... gotta change this part
            root = tree
        dimensions = root.dimensions
        # Since we can end up with many branches, keep a list of the eps budget
        # and add it all up later
        all_eps_list = []
        while self.traverse_done == False:
            if root.is_leaf:
                return ([root], all_eps_list)     
            # Splitsleft indicates when to start splitting
            # EX: splitsLeft = 3, start splitting at height-3 (actually means that there are 2 splits,
            # because the last "split" is on the leaf level and doesn't actually do anything)

            # don't split yet
            if root.height() > splitsLeft:
                root_comp = root.data[search_index]
                client_comp = self.client_val[search_index]
               
                query_result = self.compute_less(root_comp, client_comp)

                # Divide eps budget by two for EM calculation
                node_eps = eps/2
                
                DP_query_result = self.RR_query(query_result, node_eps)

                if (DP_query_result == 1):
                    if (root.right):
                        root = root.right
                elif (DP_query_result == 0):
                    root = root.left
                all_eps_list.append(node_eps*2)
                search_index = (search_index +1) % dimensions
            # start splitting
            else:
                # switch dimension to index on
                search_index = (search_index +1) % dimensions
                left_traverse, left_eps = self.traverse(root.left, client_val, eps, sensitivity, splitsLeft - 1, search_index)
                right_traverse, right_eps = self.traverse(root.right, client_val, eps, sensitivity, splitsLeft - 1, search_index)
                return (left_traverse + right_traverse, left_eps + right_eps + all_eps_list)
    
    '''
    We calculate the final epsilon assuming we implemented the mechanism with RR
    Since we can do RR with EM as long as the epsilon used for RR is half of the one used in EM
    (since EM would divide it in two), I just used a RR implementation
    '''
    def RR_query(self, query_result, eps=0.01):
        '''
        takes in a query result and add noise to the query result using random response with exponential mechanism
        '''
        eps_var = np.e ** eps
        prob_true = eps_var / (eps_var + 1)
        result_list = [query_result, 1-query_result]
        DP_result = random.choices(result_list, weights=(prob_true, 1-prob_true), k=1)[0]
        return DP_result

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def noise_LM(eps, sensitivity):
    p = random.random()
    lamb_W = sk.lambertw((p-1)/np.e, -1)
    rho = -1/(eps/sensitivity) * (lamb_W +1)
    phi = random.uniform(0, 2*np.pi)
    return pol2cart(rho, phi)

@ray.remote
def traverse_iter(max_range, database_size, eps, sensitivity, k_val, max_split, dimension, do_LDP, x_NN=1):
    '''
    k_val: number of NN to look for. (ex: k_val = 20, check if any of the results are within 20 NN of the true answer)
    '''
    # setup tree
    server = Server()
    server.setup_tree(max_range, database_size, dimension)

    # create list of true nearest neighbors
    true_result_arr_dist = server.tree.search_knn(server.client_val, k_val)
    true_result_arr = [node for (node, _) in true_result_arr_dist]
    
    # keep track of which values the methods are finding relative to the true results
    tree_true = [0] * k_val
    LDP_true = [0] * k_val
    # keep track of whether the top 5 or 10 NN have been found
    k_tree_five = 0
    k_LDP_five = 0
    k_tree_ten = 0
    k_LDP_ten= 0
    
    tree_found_results = []
    ldp_found_results = []
    
    # Traverse tree with DP method
    results, tree_eps = server.traverse(server.tree, server.client_val, eps, sensitivity, max_split)
    
    avg_num_results = len(results)
    # Bc method can return more than one result, iterate through them all
    for result in results:
        j = 0

        # this just finds the x nearest neighbors of the found value
        # not relevant to the early splitting method but a leftover from a previous experiment
        # tree_result_arr_dist = server.tree.search_knn(result[0].data, x_NN)
        # search_result_arr = [node for (node, _) in tree_result_arr_dist]

        search_result_arr = [result]
        # leftover from previous exp
        for near_result in search_result_arr:

            unique_tree = False
            # This should always be true, leftover from when there was a chance
            # that the same value could be found (prev experiment)
            if near_result not in tree_found_results:
                tree_found_results.append(near_result)
                unique_tree = True
            if (near_result in true_result_arr):
                arr_ind_tree = true_result_arr.index(near_result)
                if unique_tree:
                    tree_true[arr_ind_tree] += 1
                if (arr_ind_tree < 10):
                    k_tree_ten = 1
                if (arr_ind_tree < 5):
                    k_tree_five = 1

    # final eps calculation
    e_1 = sum(tree_eps)

    # bounded range composition
    log_val = math.log(database_size**1.1)

    first_half = 0
    sub_sqr_term = 0
    for eps in tree_eps:
        if eps != 0:
            log_term = math.log(eps/(1-np.e**(-eps)))
            first_half += ((eps/(1-np.e**(-eps))) - 1 - log_term)
        sub_sqr_term += eps**2
    sqr_term =  math.sqrt(sub_sqr_term/2 * log_val)
    final_eps = first_half + sqr_term

    # Choose the smallest eps
    if e_1 < final_eps:
        final_eps = e_1

    # since final_eps is the total eps for the tree
    # we want to run laplace len(results) times so we need
    # to divide the final_eps by the number of results
    ldp_eps = final_eps / len(results)
    
    # for 3 or more dimensions, we can't do laplace
    if do_LDP:
        
        lap_var = noise_LM(ldp_eps, sensitivity)
        LDP = tuple(map(lambda i, j: i + j, server.client_val, lap_var))
       
        LDP_result_temp_arr = server.tree.search_knn(LDP, len(results))
        LDP_result_arr = [node for (node, _) in LDP_result_temp_arr]
       
        for LDP_res in LDP_result_arr:
            unique_ldp = False
            if LDP_res not in ldp_found_results:
                ldp_found_results.append(LDP_res)
                unique_ldp = True
            if (LDP_res in true_result_arr):
                arr_ind_ldp = true_result_arr.index(LDP_res)
                if unique_ldp:
                    LDP_true[arr_ind_ldp] += 1
                if arr_ind_ldp < 10:
                    k_LDP_ten = 1
                if arr_ind_ldp < 5:
                    k_LDP_five = 1

    return(tree_true, LDP_true, k_LDP_five, k_LDP_ten, k_tree_five, k_tree_ten, len(tree_found_results), len(ldp_found_results), avg_num_results, server.tree.height(), final_eps)

def benchmark(dimension, do_LDP):
    ray.init()
    database_size = int(1e5)
    max_range = int(1E7)
    print_table = PrettyTable()
    k_val = 10
    
    with open(str(dimension) + 'DresultsRREarlySplit.csv', 'w') as f:
        write = csv.writer(f)
        headers = ["Eps/Sensitivity", "Node eps", "Sensitivity", "Tree Eps", "DB size", "Num Max Splits", "Avg Num Tree Results", "Tree results", "Avg 5 Tree", "Avg 10 Tree", "Avg Num LDP Results" ,"LDP results", "Avg 5 LDP", "Avg 10 LDP"]
        write.writerow(headers)
        print_table.field_names = headers

    while database_size <= (1*1E6):
        eps_list = [0.01, 0.1, 1,1.5, 2,3,4,5, 7, 10, 12, 17]
        sensitivity = max_range
        # in case you want to find for each tree result the x_NN nearest neighbors to that
        x_NN = 1
        for eps in eps_list:
            sensitivity = max_range
            while sensitivity > 100:
                # bc the last level doesn't actually split, subtract 1 to get split count
                max_splits = [1, 3 , 5, 9]
                for split in max_splits:
                    total_val = [0] * k_val
                    total_LDP = [0] * k_val
                    top_five_LDP = 0
                    top_ten_LDP = 0
                    top_five_tree = 0
                    top_ten_tree = 0
                    num_distinct_tree_results = 0
                    num_distinct_ldp_results = 0

                    dt = datetime.now()
                    print("Date and time is:", dt)

                    # iterations = 500
                    iterations = 50

                    
                    # parallelizes
                    results = ray.get([traverse_iter.remote(max_range, database_size, eps, sensitivity, k_val, split, dimension, do_LDP, x_NN ) for i in range(iterations)])
                    
                    final_eps = 0

                    for res in results:
                        if res[-1] > final_eps:
                            final_eps = res[-1]
                        res_x = 0
                        while res_x < len(res):
                            if res_x < 2:
                                # at [total_x] we are adding whether we found the kth nearest neighbour in this iteration
                                # res_x to keep track of each return value in return(tree_true, LDP_true, k_LDP_five, k_LDP_ten, k_tree_five, k_tree_ten, len(tree_found_results), len(ldp_found_results), avg_num_results, server.tree.height(), final_eps)
                                # so total_val[total_x] is the number of times we found the kth nearest neighbour across all iterations
                                total_x = 0
                                while total_x < len(total_val):
                                    if res_x== 0:
                                        total_val[total_x] += res[res_x][total_x]
                                    elif res_x ==1:
                                        total_LDP[total_x] += res[res_x][total_x]
                                    total_x += 1
                            else:
                                if res_x == 2:
                                    top_five_LDP += res[res_x]
                                elif res_x == 3:
                                    top_ten_LDP += res[res_x]
                                elif res_x == 4:
                                    top_five_tree += res[res_x]
                                elif res_x == 5:
                                    top_ten_tree += res[res_x]
                                elif res_x == 6:
                                    num_distinct_tree_results += res[res_x]
                                elif res_x == 7:
                                    num_distinct_ldp_results += res[res_x]
                            res_x += 1

                    # normalizes to get a percentage average
                    normalizing_factor = iterations/100
                    avg_five_LDP = top_five_LDP/normalizing_factor
                    avg_five_tree = top_five_tree/normalizing_factor
                    avg_ten_LDP = top_ten_LDP/normalizing_factor
                    avg_ten_tree = top_ten_tree/normalizing_factor
                    
                    avg_distinct_tree_results = num_distinct_tree_results/iterations
                    avg_distinct_ldp_results = num_distinct_ldp_results/iterations

                    avg_total_tree = [x/normalizing_factor for x in total_val] # why divide by normalizing factor?? turns # of times found into # of times found / (50 / 100) = 2 * number of times found
                    avg_total_LDP = [x/normalizing_factor for x in total_LDP]

                    with open(str(dimension) + 'DresultsRREarlySplit.csv', 'a') as f:
                        row = [final_eps/sensitivity, eps, sensitivity, final_eps, database_size, split-1, avg_distinct_tree_results, avg_total_tree[0:5], avg_five_tree, avg_ten_tree, avg_distinct_ldp_results, avg_total_LDP[0:5], avg_five_LDP, avg_ten_LDP]
                        write = csv.writer(f)
                        print_table.add_row(row)
                        write.writerow(row)
                    print(print_table)
                    print_table = print_table[-1]
                    total_val = [0, 0, 0, 0, 0]
                if eps == 0.1:
                    sensitivity = sensitivity / 10
                else: 
                    sensitivity = 100
                sensitivity = 100
        database_size = database_size * 10
    ray.shutdown()

benchmark(dimension=2, do_LDP=True)

