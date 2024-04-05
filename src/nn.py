# contains differentially private methods to help run the nearest neighbour algorithm
import random
import math
import numpy as np
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



# ------------------------------ Laplace ------------------------------

def sample_laplace(client, geo_eps):
    '''
    Add laplace noise then retrieve top-k nearest neighbours with a noised client
    '''

    # add noise
    if geo_eps == float('inf'):
        noised_client = client
    else:
        noised_client = list(map(lambda x, y: x + y, client, random_laplace_noise(geo_eps)))

    return noised_client



# ------------------------------ DP-TT ------------------------------


def search_dptt(client, server_tree, early_stopping_level, early_stopping_constant, sparsity_constant, scheduler):
    '''
    :client: query value
    :server_tree: the database values stored in a kdtree, with nodes pushed down for DP-TT
    :early_stopping_level: stop this number of levels before the leaf
    :early_stopping_constant: multiply the node_eps by this constant
    :sparsity_constant: to keep epsilons of different densities with the same total epsilon
    :scheduler: level -> eps

    DP-TT-CMP algorithm, returns (nearest neighbours lst, total epsilon spent)
    '''
    
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
    
    # keep track of nearest neighbours, level we're at, and stack for DFS, and total eps spent
    nn = []
    level = 0 # LEVEL MUST START AT ZERO! When constructing the kd-tree, the first split is done according to the 0th-axis. Setting level = 1 leads to comparing the 1st-axis against a node whose 0th-axis is the median, but 1st axis has no guarantees!
    queue = [server_tree]
    eps_lst = []
    eps_geo_lst = []
    dimension = len(client)

    while queue:
        node = queue.pop()
        
        if not node:
            continue

        if node.is_leaf:
            nn.append(node)
            continue
        
        if node.height() > early_stopping_level:
            axis = level % dimension

            node_eps = sparsity_constant * early_stopping_constant * scheduler(level)
            noised_choose_right = apply_exponential_mechanism_cmp(client, node, axis, node_eps)

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

def sample_lsrr(client, server_tree, eps, domain):
    '''
    Implementation of L-SRR mechanism: https://arxiv.org/pdf/2209.15091.pdf

    Assumes that number of groups same as the height of the encoding tree.

    To sample a point using L-SRR, we
        0. Calculate constants
        1. Sample a group
        2. Sample a point from given group in a unit square
        3. Scale and translate the unit square to the domain (i.e., a rectangle)
    
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
    
    def sample_point_within_group(group, client):
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
        unit_length = 1 / 2 ** ((m - 1) - group)
        # which square is the client in?
        bounds = calculate_bounds(client, unit_length)
        
        # sample client from the Group's bounds with equal probability
        noised_client = [random.uniform(lower, upper) for (lower, upper) in bounds]
        
        # if we didn't sample G_0 (i.e., if we have a "missing square")
        if group > 0:
            
            prev_unit_length = 1 / 2 ** ((m - 1) - (group - 1))
            prev_bounds = calculate_bounds(client, prev_unit_length)

            # sample until no longer in previous group
            while is_within_bounds(noised_client, prev_bounds):
                noised_client =  [random.uniform(lower, upper) for (lower, upper) in bounds]

        return noised_client


    # 0. Calculate constants
    dimension = len(client)
    
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

    
    # 2. Sample a point uniformly from given group in a rectangle
    client_scaled_to_unit_square = [(val - lower) / (upper - lower) for val, (lower, upper) in zip(client, domain)]
    noised_client_scaled_to_unit_square = sample_point_within_group(group, client_scaled_to_unit_square)

    
    # 3. Transform the unit square to the domain
    noised_client = [lower + val * (upper - lower) for val, (lower, upper) in zip(noised_client_scaled_to_unit_square, domain)]

    return noised_client


# ------------------------------ Square Mechanism ------------------------------

def truncate(data, min_, max_):
    """
    Args:
        data: data to truncate
        min_: the minimum value used for truncation
        max_: the maximum value used for truncation
    Returns:
        The truncated data where each value lie between input `min_` and input `max_`
    """
    return np.maximum(np.minimum(max_, data), min_)


def compute_w_sm(eps, domain):
    '''
    Compute optimal square side length by solving equation (9)
    '''
    l_x = domain[0][1] - domain[0][0]
    l_y = domain[1][1] - domain[1][0]

    coeff_list = np.zeros(6)
    l_x_square = l_x ** 2
    l_y_square = l_y ** 2
    eps_term = np.exp(eps) - 1
    coeff_list[0] = -4 * l_x_square * l_y_square * (l_x_square + l_y_square)
    coeff_list[2] = 8 * l_x_square * l_y_square
    coeff_list[3] = 5 * l_x * l_y * (l_x + l_y)
    coeff_list[4] = 4 * l_x * l_y * eps_term
    coeff_list[5] = 3 * eps_term * (l_x + l_y)
    coeff_list = np.flip(coeff_list)
    roots = np.roots(coeff_list)
    real_roots = np.compress(np.isreal(roots), roots)
    pos_roots = np.compress(real_roots > 0, real_roots)
    assert len(pos_roots) == 1
    return truncate(np.real(pos_roots[0]), 0, min(l_x, l_y))


def compute_alpha_sm(eps, domain, w):
    '''
    Compute alpha. Note that w represents the optimal side length computed above
    '''
    l_x = domain[0][1] - domain[0][0]
    l_y = domain[1][1] - domain[1][0]
    return 1 / (l_x * l_y + (np.exp(eps) - 1) * w ** 2)
    

def calc_square_center(client, domain, w):
    """
    :param t: the input data to perturb
    :param out_domain: the output domain.
    :param b: the side length of the square region
    :return: the centers of the square regions
    """
    domain_u_bound = np.array([domain[0][1], domain[1][1]]).reshape(1, 2)
    domain_l_bound = np.array([domain[0][0], domain[1][0]]).reshape(1, 2)
    center_upper_bound = domain_u_bound - w / 2
    center_lower_bound = domain_l_bound + w / 2
    if not np.all(center_upper_bound - center_lower_bound >= 0):
        print('t = {}'.format(client))
        print('out_domain = {}'.format(domain))
        print('side_length = {}'.format(w))
        print('domain_u_bound = {}'.format(domain_u_bound))
        print('domain_l_bound = {}'.format(domain_l_bound))
        print('center_upper_bound = {}'.format(center_upper_bound))
        print('center_lower_bound = {}'.format(center_lower_bound))
        raise Exception('not all(center_upper_bound - center_lower_bound >= 0)')
    center = np.array(client).reshape(1, 2)
    center = np.minimum(center, center_upper_bound)
    center = np.maximum(center, center_lower_bound)
    return center


def sample_sm(client, eps, domain):
    '''
    An implementation of the square mechanism paper(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9458926)'s Algorithm 1

    1. Calculate optimal side length and alpha (done once for a domain)
    2. Compute the center of the square
    3. Sample x from [0, 1]
    4. if x ≤ 4αb1b2 = αl1l2, sample uniformly from D
    5. otherwise, sample from the square
    '''

    w = compute_w_sm(eps, domain)
    alpha = compute_alpha_sm(eps, domain, w)
    center = calc_square_center(client, domain, w)

    square_sample = np.random.uniform(low=-w/2, high=w/2, size=center.shape) + center
    domain_sample = np.random.uniform(low=np.array([domain[0][0], domain[1][0]]), high=np.array([domain[0][1], domain[1][1]]), size=center.shape)

    is_domain_sample = np.random.binomial(n=1, p=alpha * (domain[0][1] - domain[0][0]) * (domain[1][1] - domain[1][0]))
    final_sample =  np.where(is_domain_sample, domain_sample, square_sample)
    return list(final_sample)[0]

