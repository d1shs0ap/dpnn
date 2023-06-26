# contains differentially private methods to help run the nearest neighbour algorithm
import random
import math
import scipy
# from GeoPrivacy.mechanism import random_laplace_noise


# ------------------------------ TRUE NN ------------------------------

def search_true(client, server_tree, k):
    '''
    Retrieve the true top-k nearest neighbours
    '''
    true_nn_and_dists = server_tree.search_knn(client, k) # if distance is not given, Euclidean distance is assumed
    true_nn = list(map(lambda node_and_dist: node_and_dist[0], true_nn_and_dists))
    return true_nn



# ------------------------------ LDP ------------------------------

def search_laplace(client, server_tree, dimension, geo_eps, k):
    '''
    Add laplace noise then retrieve top-k nearest neighbours with a noised client
    '''
    
    def generate_laplace_noise(dimension, geo_eps):
        '''
        Helper function to generate Laplace noise
        '''
        def generate_planar_laplace_noise(geo_eps):
            '''
            :geo_eps: for eps-geo-indistinguishability, *not* for eps-privacy

            Generates laplace that guarantees eps-geo-indistinguishability. Guarantee provided by Theorem 4.1 (if we ignore precision and truncation, first term of RHS is the bound) of geo-indistinguishability paper
            '''
            # generate polar coordinates
            theta = random.uniform(0, 2 * math.pi) # angle

            # radius, using the method outlined in the geo-indistinguishability paper
            p = random.uniform(0, 1)
            r = -1 / geo_eps * (scipy.special.lambertw((p - 1) / math.e, k=-1).real + 1)

            # convert polar coordinates to cartesian coordinates
            x, y = r * math.cos(theta), r * math.sin(theta)

            return x, y

        if dimension == 2:
            return generate_planar_laplace_noise(geo_eps)

        else:
            raise Exception("dimension must be 2")

    # use LDP to retrieve top NN
    noised_client = list(map(lambda x, y: x + y, client, generate_laplace_noise(dimension, geo_eps)))
    ldp_nn_and_dists = server_tree.search_knn(noised_client, k)
    ldp_nn = list(map(lambda node_and_dist: node_and_dist[0], ldp_nn_and_dists))

    return ldp_nn



# ------------------------------ DPTT ------------------------------

def apply_exponential_mechanism_cmp(client, node, axis, eps):
    '''
    exponential mechanism for selecting 0 or 1
    '''
    # the correct choice will be 1 if client is to the right of node, 0 o/w.
    correct_choice = client[axis] > node.data[axis]

    # probability that the bit given is unchanged
    stay_with_correct_choice = math.exp(eps / 2)

    # choose bit with exponential mechanism
    chosen_bit = random.choices([correct_choice, not correct_choice], weights=[stay_with_correct_choice, 1])
    return chosen_bit[0]

def apply_exponential_mechanism_dis(client, node, axis, geo_eps):
    '''
    exponential mechanism for selecting based on distance from reference point
    '''
    # probability that the positive, negative direction are chosen
    choose_left_prob = math.exp(geo_eps / 2 * (node.data[axis] - client[axis]))
    choose_right_prob = math.exp(geo_eps / 2 * (client[axis] - node.data[axis]))

    # choose bit with exponential mechanism
    chosen_bit = random.choices([0, 1], weights=[choose_left_prob, choose_right_prob])
    return chosen_bit[0]

def search_dptt(client, server_tree, dimension, early_stopping_level, node_geo_eps_generator, cmp, sensitivity):
    '''
    :client: query value
    :server_tree: the database values stored in a kdtree
    :dimension: dimension of query / server value
    :early_stopping_level: stop this number of levels before the leaf
    :node_geo_eps_generator: level -> eps

    DP-TT-CMP algorithm, returns (nearest neighbours lst, total epsilon spent)
    '''
    # keep track of nearest neighbours, level we're at, and stack for DFS, and total eps spent
    nn = []
    level = 0 # LEVEL MUST START AT ZERO! When constructing the kd-tree, the first split is done according to the 0th-axis. Setting level = 1 leads to comparing the 1st-axis against a node whose 0th-axis is the median, but 1st axis has no guarantees!
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
            node_geo_eps = node_geo_eps_generator(level)
            eps_lst.append(node_geo_eps)

            # -------------------------------- DP-TT CMP --------------------------------
            if cmp:
                noised_choose_right = apply_exponential_mechanism_cmp(client, node, axis, eps=node_geo_eps * sensitivity)
            
            # -------------------------------- DP-TT DIS --------------------------------
            else:
                noised_choose_right = apply_exponential_mechanism_dis(client, node, axis, geo_eps=node_geo_eps)

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

