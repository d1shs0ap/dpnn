# contains differentially private methods to help run the nearest neighbour algorithm
import random
import math
import scipy

def generate_laplace_noise(dimension, eps, sensitivity):

    def generate_planar_laplace_noise(eps_geo_ind):
        '''
        :eps_geo_ind: for eps-geo-indistinguishability, *not* for eps-privacy

        Generates laplace that guarantees eps-geo-indistinguishability. Guarantee provided by Theorem 4.1 (if we ignore precision and truncation, first term of RHS is the bound) of geo-indistinguishability paper
        '''
        # generate polar coordinates
        theta = random.uniform(0, 2 * math.pi) # angle

        # radius, using the method outlined in the geo-indistinguishability paper
        p = random.uniform(0, 1)
        r = -1 / eps_geo_ind * (scipy.special.lambertw((p - 1) / math.e, k=-1).real + 1)

        # convert polar coordinates to cartesian coordinates
        x, y = r * math.cos(theta), r * math.sin(theta)

        return x, y

    if dimension == 1:
        return scipy.stats.laplace.rvs(loc=0, scale=sensitivity / eps)
    
    elif dimension == 2:
        # eps-DP is equivalent to eps-privacy, or (eps/r)-geo-indistinguishability; but the function gives eps -> eps-geo-indistinguishability
        return generate_planar_laplace_noise(eps / sensitivity)
    
    else:
        raise Exception("dimension must be 1 or 2")


def search_dptt_cmp(server_tree, client, node_eps, splits_left=0):
    '''

    DP-TT-CMP algorithm
    '''

    def apply_exponential_mechanism(bit, eps):
        # probability that the bit given is unchanged
        bit_unchanged_prob = math.exp(eps / 2)

        # choose bit with exponential mechanism
        chosen_bit = random.choices([bit, not bit], weights=[bit_unchanged_prob, 1])
        return chosen_bit

    # keep track of nearest neighbours and total epsilon used up    
    nn = []

    # stack for dfs and axis for kdtree
    queue = [server_tree]
    axis = 0
    
    while queue:
        node = queue.pop()
        
        if not node:
            continue
        if node.is_leaf:
            nn.append(node)
            continue
        
        # early stopping
        if splits_left > 0:
            # if client data greater than current node, choose right child (the bit below will be 1)
            choose_right = client[axis] > node.data[axis]
            noised_choose_right = apply_exponential_mechanism(choose_right, node_eps)

            # traverse down the exp-mechanism-randomized path
            if noised_choose_right:
                queue.append(node.right)
            else:
                queue.append(node.left)
            
            splits_left -= 1
            axis = 1 - axis
            continue

        # after we get past early stopping, add everything
        nn.append(node)
        queue.append(node.left)
        queue.append(node.right)
    
    return nn
