# server generation
DIMENSION = 2
SERVER_SIZE = 10 ** 2 # 1 gas station per 1km x 1km square, lead
SERVER_DOMAIN = 10 ** 5 # 100km by 100km radius, 20 times Toronto

# client generation
CLIENT_BATCH_SIZE = 100 # number of different client value to try per server
CLIENT_SENSITIVITY = 10 ** 5 # 100km by 100km radius

# precision and recall considers top K values relevant
TRUE_K = 10

# DP-TT
EARLY_STOPPING_LEVELS = [1, 3, 5] # the level at which we stop splitting (e.g., 3 -> split thrice at level 0, 1, 2)

# functions used to determine node epsilon for exp. mech. as we move down the levels (first level starts at 0)
GEO_EPS_UNIT = 0.0000001
NODE_GEO_EPS_GENERATORS = [
    lambda level: 1 * GEO_EPS_UNIT,
    lambda level: 10 * GEO_EPS_UNIT,
    lambda level: 100 * GEO_EPS_UNIT,
    lambda level: 150 * GEO_EPS_UNIT,
    lambda level: 200 * GEO_EPS_UNIT,
    lambda level: 300 * GEO_EPS_UNIT,
    lambda level: 400 * GEO_EPS_UNIT,
    lambda level: 500 * GEO_EPS_UNIT,
    lambda level: 700 * GEO_EPS_UNIT,
    lambda level: 1000 * GEO_EPS_UNIT,
    # lambda level: 0.000000001,
    # lambda level: 0.000000005,
    # lambda level: 0.00000001,
    # lambda level: 0.00000005,
    # lambda level: 0.0000001,
    # lambda level: 0.0000005,
    # lambda level: 0.000001,
]


# NODE_EPS_GENERATORS = [
#     lambda level: 0.01 * 5.3 / (level ** 2),
#     lambda level: 0.1 * 5.3 / (level ** 2),
#     lambda level: 1 * 5.3 / (level ** 2),
#     lambda level: 1.5 * 5.3 / (level ** 2),
#     lambda level: 2 * 5.3 / (level ** 2),
#     lambda level: 3 * 5.3 / (level ** 2),
#     lambda level: 4 * 5.3 / (level ** 2),
#     lambda level: 5 * 5.3 / (level ** 2),
#     lambda level: 7 * 5.3 / (level ** 2),
#     lambda level: 10 * 5.3 / (level ** 2),
#     lambda level: 12 * 5.3 / (level ** 2),
#     lambda level: 17 * 5.3 / (level ** 2)
# ]

LSRR_M = 10 # separate the map into a tree of height 23

OUTPUT_DIR = 'graphs/lsrr'