# contains differentially private methods to help run the nearest neighbour algorithm
import random
import math
import scipy
from GeoPrivacy.mechanism import random_laplace_noise


# ------------------------------ TRUE NN ------------------------------


def search_true(client, server_tree, k):
    '''
    Retrieve the true top-k nearest neighbours
    '''
    true_nn_and_dists = server_tree.search_knn(client, k) # if distance is not given, Euclidean distance is assumed
    true_nn = list(map(lambda node_and_dist: node_and_dist[0], true_nn_and_dists))
    return true_nn

def search_within_radius(client, server_tree, radius):
    '''
    Retrieve all neighbours within a given radius
    '''
    nn_within_radius = server_tree.search_nn_dist(client, radius) # if distance is not given, Euclidean distance is assumed
    return nn_within_radius



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
    noised_client = list(map(lambda x, y: x + y, client, random_laplace_noise(geo_eps)))
    ldp_nn_and_dists = server_tree.search_knn(noised_client, k)
    ldp_nn = list(map(lambda node_and_dist: node_and_dist[0], ldp_nn_and_dists))

    return ldp_nn



# ------------------------------ DP-TT ------------------------------

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

def search_dptt(client, server_tree, dimension, early_stopping_level, scheduler, cmp):
    '''
    :client: query value
    :server_tree: the database values stored in a kdtree, with nodes pushed down for DP-TT
    :dimension: dimension of query / server value
    :early_stopping_level: stop this number of levels before the leaf
    :scheduler: level -> eps

    DP-TT-CMP algorithm, returns (nearest neighbours lst, total epsilon spent)
    '''
    # keep track of nearest neighbours, level we're at, and stack for DFS, and total eps spent
    nn = []
    level = 0 # LEVEL MUST START AT ZERO! When constructing the kd-tree, the first split is done according to the 0th-axis. Setting level = 1 leads to comparing the 1st-axis against a node whose 0th-axis is the median, but 1st axis has no guarantees!
    queue = [server_tree]
    eps_lst = []
    eps_geo_lst = []

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

            # -------------------------------- DP-TT CMP --------------------------------
            # if cmp:
            node_eps = scheduler(level)
            noised_choose_right = apply_exponential_mechanism_cmp(client, node, axis, eps=node_eps)
            
            # # -------------------------------- DP-TT DIS --------------------------------
            # else:
            #     node_eps = scheduler(level) / abs(node.data[axis] - client[axis])
            #     noised_choose_right = apply_exponential_mechanism_dis(client, node, axis, geo_eps=node_eps / sensitivity)

            # traverse down the exp-mechanism-randomized path
            if noised_choose_right:
                queue.append(node.right)
            else:
                queue.append(node.left)

            eps_lst.append(node_eps)
            eps_geo_lst.append(node_eps / (2 * abs(node.data[axis] - client[axis])))
            
            level += 1

            continue

        # after we get past early stopping, add everything
        queue.append(node.left)
        queue.append(node.right)
    
    return nn, eps_lst, eps_geo_lst


# ------------------------------ L-SRR ------------------------------

def search_lsrr(client, server_tree, dimension, eps, k, domain):
    '''
    Implementation of L-SRR mechanism: https://arxiv.org/pdf/2209.15091.pdf

    Assumes that number of groups same as the height of the encoding tree.

    To sample a point using L-SRR, we
        0. Calculate constants
        1. Sample a group
        2. Sample a point uniformly from given group
    
    Note that here we don't calculate the exact group size but instead relative group sizes, treating |G_1| = 1
    '''

    def calculate_a_min(c, d):
        '''
        Equation (4) in L-SRR paper
        '''
        sum_term = 0
        for i in range(1, m):
            group_size = (2 ** dimension - 1) * ((2 ** (i - 1)) ** dimension)
            sum_term += i * group_size

        # a_min and a_max
        a_min = (m - 1) / ((m - 1) * d * c - (c - 1) * sum_term)

        return a_min

    def sample_group(a_max, delta):
        '''
        Sample a group in the staircase mechanism,
        
        Equation (2) in L-SRR paper
        '''
        groups = [i for i in range(m)]

        group_weights = [a_max * 1] # a_1 * |G_1|
        for i in range(1, m): # rest of the groups
            a_i = a_max - i * delta
            group_size = (2 ** dimension - 1) * ((2 ** (i - 1)) ** dimension)
            
            # the weight of each group is alpha_i * |G_i|
            group_weights.append(a_i * group_size)

        selected_group = random.choices(groups, weights=group_weights)
        return selected_group[0]
    
    def sample_point_within_group(group):
        '''
        Given the selected group, sample a point uniformly within the group
        '''
        def calculate_bounds(point, unit_length):
            '''
            if the map were to be divided into squares of unit_length, which square does our client belong in?
            '''
            bounds = []
            for dim in range(dimension):
                unit_number = point[dim] // unit_length # the index of the largest unit smaller than the client on the given dimension
                lower, upper = unit_number * unit_length, (unit_number + 1) * unit_length
                
                bounds.append((lower, upper))
            
            return bounds

        
        def is_within_bounds(point, bounds):
            '''
            Checks if point is within given square
            '''
            for dim, (lower, upper) in enumerate(bounds):
                if lower <= point[dim] <= upper:
                    continue
                return False # as long as one axis is not within bounds, client not within bounds
            return True
        
        # one "unit" in this group = domain / # of pieces we are splitting the domain into
        unit_length = domain / 2 ** ((m - 1) - group)
        bounds = calculate_bounds(client, unit_length)
        
        # sample client from the Group's bounds with equal probability
        noised_client = [random.uniform(lower, upper) for (lower, upper) in bounds]
        
        # if previous group exists
        if group > 0:
            
            prev_unit_length = domain / 2 ** ((m - 1) - (group - 1))
            prev_bounds = calculate_bounds(client, prev_unit_length)

            # sample until no longer in previous group
            while is_within_bounds(noised_client, prev_bounds):
                noised_client =  [random.uniform(lower, upper) for (lower, upper) in bounds]

        return noised_client

    # 0. Calculate constants
    if dimension == 1:
        m = round(math.log2(server_tree.height()))
    elif dimension == 2:
        m = max(round(math.log2(server_tree.height()) / 2), 2) # divide by two because each 2D splits into four groups

    c = math.exp(eps) # Theorem 3.3, when group size does not change based on location (x and x')
    d = (2 ** (m - 1)) ** dimension

    a_min = calculate_a_min(c, d)
    a_max = c * a_min
    delta = (a_max - a_min) / (m - 1) # (3) in L-SRR paper
    
    # 1. Sample a group
    group = sample_group(a_max, delta)

    # 2. Sample a point uniformly from given group
    noised_client = sample_point_within_group(group)

    # NN search using the noised point
    lsrr_nn_and_dists = server_tree.search_knn(noised_client, k)
    lsrr_nn = list(map(lambda node_and_dist: node_and_dist[0], lsrr_nn_and_dists))

    return lsrr_nn