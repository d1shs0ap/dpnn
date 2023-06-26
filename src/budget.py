import math

def calculate_adaptive_eps(eps_lst, delta):
    '''
    :eps_lst: list of epsilons used in the adaptive composition
    :delta: used in the calculation formula for (eps, delta)-DP, set to at most 1 / server size
    
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

def calculate_eps_sum(eps_lst):
    return sum(eps_lst)