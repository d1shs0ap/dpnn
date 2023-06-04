import math

def calculate_adaptive_eps(eps_lst, delta=1):
    '''
    :eps_lst: list of epsilons used in the adaptive composition
    :delta: used in the calculation formula for (eps, delta)-DP
    
    Calculates the bound provided in Proposition 4 of http://proceedings.mlr.press/v119/dong20a/dong20a.pdf
    '''

    # calculating the two parts of the second term of the minimum
    first_part, second_part = 0, 0
    
    for eps_i in eps_lst:
        frac = eps_i / (1 - math.exp(-eps_i))
        first_part += frac - 1 - math.log(frac)

        second_part += eps_i ** 2
    second_part = math.sqrt(1/2 * second_part * math.log(1/delta))
    
    second_term = first_part + second_part

    final_eps = min(sum(eps_lst), second_term)
    return final_eps

def calculate_eps_to_node_eps(node_eps_functions, early_stopping_level):
    '''
    :node_eps_functions: a list of functions, each calculating level -> epsilon at level
    :early_stop: when to stop splitting

    :return: {total eps -> [node eps]}

    Calculate the list of node epsilons that we will be using as we move down the tree, given different types of formulas
    '''
    eps_to_node_eps = {}

    for f in node_eps_functions:
        
        # calculate node epsilons at each level
        node_eps = []
        for level in range(early_stopping_level):
            node_eps_at_level = f(level)
            node_eps.append(node_eps_at_level)

        # calculate total eps
        eps = calculate_adaptive_eps(node_eps)
        eps_to_node_eps[eps] = node_eps

    return eps_to_node_eps
