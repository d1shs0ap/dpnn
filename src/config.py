# for random geneartion of server and client
DIMENSION = 2
SERVER_SIZES = [10 ** 5]
CLIENT_BATCH_SIZE = 10 # number of different client value to try per server

SENSITIVITIES = [10 ** 7]

# DP NN search parameters
TRUE_K = 10

# LDP
LDP_K = 25000

# DP-TT
EARLY_STOPPING_LEVEL = 2
# functions used to determine node epsilon for exp. mech. as we move down the levels (first level starts at 0)
NODE_EPS_FUNCTIONS = [lambda level: 0.1, lambda level: 20, lambda level: 10 / (level + 1)]

