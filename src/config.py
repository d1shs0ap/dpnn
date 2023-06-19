# server generation
DIMENSION = 2
SERVER_SIZE = 10 ** 5
SERVER_DOMAIN = 10 ** 7

# client generation
CLIENT_BATCH_SIZE = 500 # number of different client value to try per server
CLIENT_SENSITIVITY = 10 ** 7

# precision and recall considers top K values relevant
TRUE_K = 10

# DP-TT
EARLY_STOPPING_LEVELS = [1, 3, 5, 9] # the level at which we stop splitting (e.g., 3 -> split thrice at level 0, 1, 2)

# functions used to determine node epsilon for exp. mech. as we move down the levels (first level starts at 0)
# NODE_EPS_GENERATORS = [
#     lambda level: 0.01,
#     lambda level: 0.1,
#     lambda level: 1,
#     lambda level: 1.5,
#     lambda level: 2,
#     lambda level: 3,
#     lambda level: 4,
#     lambda level: 5,
#     lambda level: 7,
#     lambda level: 10,
#     lambda level: 12,
#     lambda level: 17
# ]


NODE_EPS_GENERATORS = [
    lambda level: 0.01 * 5.3 / (level ** 2),
    lambda level: 0.1 * 5.3 / (level ** 2),
    lambda level: 1 * 5.3 / (level ** 2),
    lambda level: 1.5 * 5.3 / (level ** 2),
    lambda level: 2 * 5.3 / (level ** 2),
    lambda level: 3 * 5.3 / (level ** 2),
    lambda level: 4 * 5.3 / (level ** 2),
    lambda level: 5 * 5.3 / (level ** 2),
    lambda level: 7 * 5.3 / (level ** 2),
    lambda level: 10 * 5.3 / (level ** 2),
    lambda level: 12 * 5.3 / (level ** 2),
    lambda level: 17 * 5.3 / (level ** 2)
]