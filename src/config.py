import math

# server generation
DIMENSION = 2

SERVER_BATCH_SIZE = 1
SERVER_SIZE = 10 ** 5 # 100, 000 people living here
SERVER_DOMAIN = math.sqrt(10 ** 7) # 10 km^2 total area

# client generation
CLIENT_BATCH_SIZE = 500 # number of different client value to try per server
CLIENT_SENSITIVITY = SERVER_DOMAIN

# precision and recall considers all values within 5km to be a true nearest neighbour
TRUE_RADIUS_SQUARED = (SERVER_DOMAIN ** 2 / SERVER_SIZE) * 10

# DP-TT
# EARLY_STOPPING_LEVELS = [5]
EARLY_STOPPING_LEVELS = [1, 3, 5, 9] # the level at which we stop splitting (e.g., 3 -> split thrice at level 0, 1, 2)

# functions used to determine node epsilon for exp. mech. as we move down the levels (first level starts at 0)
# EPSILONS = [0.01, 1, 2, 5, 7, 12, 17]
EPSILONS = [0.01, 0.1, 1, 1.5, 2, 3, 4, 5, 7, 10, 12, 17]

def constant(eps):
    return lambda level: eps

def linear(eps):
    return lambda level: 1.5 * eps - 0.07 * eps * (level + 1)

def log(eps):
    return lambda level: 1.5 * eps - 0.28 * eps * math.log(level + 1)

def sqrt(eps):
    return lambda level: 1.5 * eps - 0.2 * eps * math.sqrt(level + 1)

def quadratic(eps):
    return lambda level: 1.5 * eps - 0.008 * eps * (level + 1) ** 2

# show increasing epsilon is bad, when pronounced
def reverse_linear(eps):
    height = int(math.log2(SERVER_SIZE)) - 2
    return lambda level: (1.6 - 0.084 * height) * eps + 0.084 * eps * (level + 1)

# show decreasing too fast is bad
def inverse_sqrt(eps):
    return lambda level: 2.15 * eps / (level + 1) ** (1 / 2)

def inverse_linear(eps):
    return lambda level: 4 * eps / (level + 1)

def inverse_quadratic(eps):
    return lambda level: 8 * eps / (level + 1) ** 2

def best_scheduler(eps):
    return lambda level: 1.3 * eps - 0.17 * eps * math.sqrt(level + 1)

SCHEDULER_TYPES = [constant]
# SCHEDULER_TYPES = [constant, linear, log, sqrt, quadratic, reverse_linear, inverse_sqrt, inverse_linear, inverse_quadratic]
SCHEDULER_TYPE_TO_SCHEDULERS = {scheduler_type.__name__: [scheduler_type(eps) for eps in EPSILONS] for scheduler_type in SCHEDULER_TYPES}

OUTPUT_DIR = f'graphs/densities/size_{SERVER_SIZE}_domain_{int(SERVER_DOMAIN ** 2)}'