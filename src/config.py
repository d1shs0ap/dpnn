# for random geneartion of server and client
DIMENSION = 2
SERVER_SIZE = 10 ** 5

LOWER_BOUND = 0
UPPER_BOUND = 10 ** 7

# DP NN search parameters
K = 10

# LDP
LDP_K = 25000
LDP_EPS = 0.1 # the epsilon that guarantees eps-privacy, *not* eps-geo-indistinguishability because eps-privacy is the equivalent analog for eps-DP

# DP-TT
DPTT_NODE_EPS = 0.1
EARLY_STOPPING_LEVEL = 2
