# contains differentially private methods to help run the nearest neighbour algorithm
import random
import math
import scipy


# ------------------------------ TRUE NN ------------------------------

def search_true(client, server_tree, k):
    '''
    Retrieve the true top-k nearest neighbours
    '''
    true_nn_and_dists = server_tree.search_knn(client, k) # if distance is not given, Euclidean distance is assumed
    true_nn = list(map(lambda node_and_dist: node_and_dist[0], true_nn_and_dists))
    return true_nn



# ------------------------------ LDP ------------------------------

def search_ldp(client, server_tree, dimension, eps, sensitivity, k):
    '''
    LDP algorithm: etrieve top-k nearest neighbours with a noised client
    '''
    
    def generate_laplace_noise(dimension, eps, sensitivity):
        '''
        Helper function to generate Laplace noise
        '''
        def generate_planar_laplace_noise(eps_geo):
            '''
            :eps_geo: for eps-geo-indistinguishability, *not* for eps-privacy

            Generates laplace that guarantees eps-geo-indistinguishability. Guarantee provided by Theorem 4.1 (if we ignore precision and truncation, first term of RHS is the bound) of geo-indistinguishability paper
            '''
            # generate polar coordinates
            theta = random.uniform(0, 2 * math.pi) # angle

            # radius, using the method outlined in the geo-indistinguishability paper
            p = random.uniform(0, 1)
            r = -1 / eps_geo * (scipy.special.lambertw((p - 1) / math.e, k=-1).real + 1)

            # convert polar coordinates to cartesian coordinates
            x, y = r * math.cos(theta), r * math.sin(theta)

            return x, y

        if dimension == 1:
            return scipy.stats.laplace.rvs(loc=0, scale=sensitivity / eps)
        
        elif dimension == 2:
            # eps-DP is equivalent to eps-privacy, or (eps/r)-geo-indistinguishability; but the function gives eps -> eps-geo-indistinguishability
            return generate_planar_laplace_noise(eps_geo=eps / sensitivity)

        else:
            raise Exception("dimension must be 1 or 2")

    # use LDP to retrieve top NN
    noised_client = list(map(lambda x, y: x + y, client, generate_laplace_noise(dimension, eps, sensitivity)))
    ldp_nn_and_dists = server_tree.search_knn(noised_client, k)
    ldp_nn = list(map(lambda node_and_dist: node_and_dist[0], ldp_nn_and_dists))

    return ldp_nn



# ------------------------------ DPTT ------------------------------

def apply_exponential_mechanism(bit, eps):
    '''
    exponential mechanism for selecting 0 or 1
    '''
    # probability that the bit given is unchanged
    bit_unchanged_prob = math.exp(eps / 2)

    # choose bit with exponential mechanism
    chosen_bit = random.choices([bit, not bit], weights=[bit_unchanged_prob, 1])
    return chosen_bit[0]

def search_dptt_cmp(client, server_tree, dimension, early_stopping_level, node_eps_generator):
    '''
    :client: query value
    :server_tree: the database values stored in a kdtree
    :dimension: dimension of query / server value
    :early_stopping_level: stop this number of levels before the leaf
    :node_eps_generator: level -> eps

    DP-TT-CMP algorithm, returns (nearest neighbours lst, total epsilon spent)
    '''
    # keep track of nearest neighbours, level we're at, and stack for DFS, and total eps spent
    nn = []
    level = 1
    queue = [server_tree]
    eps_lst = []

    while queue:
        node = queue.pop()
        
        if not node:
            continue

        if node.is_leaf:
            nn.append(node)
            continue
        
        # tree w/. one node has height = 1, so if early stopping level = 1, we split at second last layer
        if node.height() > early_stopping_level:
            axis = level % dimension

            # generate and record node eps
            node_eps = node_eps_generator(level)
            eps_lst.append(node_eps)

            # if client data greater than current node, choose right child (the bit below will be 1)
            choose_right = client[axis] > node.data[axis]
            noised_choose_right = apply_exponential_mechanism(choose_right, node_eps)

            # traverse down the exp-mechanism-randomized path
            if noised_choose_right:
                queue.append(node.right)
            else:
                queue.append(node.left)
            
            level += 1
            continue

        # after we get past early stopping, add everything
        queue.append(node.left)
        queue.append(node.right)
    
    return nn, eps_lst

