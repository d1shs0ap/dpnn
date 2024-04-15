import math
from schedulers import *
from data import Dataset

class Config:
    def __init__(
        self,
        domain = [(0, 10 ** 3), (0, 10 ** 3)],
        server_size = 10 ** 5,
        client_batch_size = 500,
        early_stopping_levels = {1: 1, 3: 1, 5: 1, 7: 1},
        epsilons = [3.5 + .8 * i for i in range(8)],
        laplace_epsilons = [18 + 25 * i for i in range(8)],
        lsrr_epsilons = [1, 2, 3, 4],
        sm_epsilons = [5 + 1.5 * i for i in range(8)],
        scheduler_types = [constant],
        experiment_series = 'densities',
        dataset = Dataset.RANDOM,
        true_radius_constant = 10,
        sparsity_constant = 1,
    ):
        self.domain = domain
        self.server_size = server_size
        self.client_batch_size = client_batch_size
        self.early_stopping_levels = early_stopping_levels
        self.epsilons = epsilons
        self.scheduler_types = scheduler_types
        self.experiment_series = experiment_series
        self.dataset = dataset
        self.true_radius_constant = true_radius_constant
        self.sparsity_constant = sparsity_constant
        self.laplace_epsilons = laplace_epsilons
        self.lsrr_epsilons = lsrr_epsilons
        self.sm_epsilons = sm_epsilons

    @property
    def client_sensitivity(self):
        return math.sqrt(sum([(upper - lower) ** 2 for (lower, upper) in self.domain]))
    
    @property
    def server_area(self):
        return math.prod([upper - lower for (lower, upper) in self.domain])

    @property
    def true_radius_squared(self):
        return self.server_area / self.server_size * self.true_radius_constant
    
    @property
    def output_dir(self):
        if self.experiment_series == 'densities':
            return f'graphs/densities/{self.server_size}'
        elif self.experiment_series == '1d':
            return f'graphs/1d/{self.server_size}'
        else:
            return f'graphs/{self.experiment_series}'

# SCHEDULER_TYPES = [constant, linear, log, sqrt, quadratic, reverse_linear, inverse_sqrt, inverse_linear, inverse_quadratic]

density_config_2 = Config(server_size = 10 ** 2, early_stopping_levels={1: 1, 3: 1.17, 5: 1.39, }, sparsity_constant = 2, epsilons=[0.5, 1, 2], domain = [(0, math.sqrt(10 ** 10)), (0, math.sqrt(10 ** 10))])
density_config_3 = Config(server_size = 10 ** 3, early_stopping_levels={4: 1.25, 5: 1.3, 6: 1.5, 7: 2}, sparsity_constant = 2, epsilons=[0.1, 0.3, 0.7, 0.9])
density_config_4 = Config(server_size = 10 ** 4, early_stopping_levels={1: 1, 3: 1.17, 5: 1.39, 9: 2.32}, sparsity_constant = 1.44)
# density_config_5 = Config(server_size = 10 ** 5, early_stopping_levels={1: 1, 3: 1.14, 5: 1.3, 9: 1.9}, sparsity_constant = 1.17, client_batch_size=20)
# density_config_6 = Config(server_size = 10 ** 6, early_stopping_levels={1: 1, 3: 1.11, 5: 1.25, 9: 1.67}, sparsity_constant = 1, client_batch_size=20)
density_config_5 = Config(server_size = 10 ** 5, early_stopping_levels = {1: 1, 3: 1, 5: 1, 7: 1}, epsilons = [3.5 + .8 * i for i in range(8)], laplace_epsilons = [18 + 25 * i for i in range(8)], sm_epsilons = [5 + 1.5 * i for i in range(8)],client_batch_size=500)
density_config_6 = Config(server_size = 10 ** 6, early_stopping_levels={1: 1}, sparsity_constant = 1, client_batch_size=20)

gowalla_sf_config = Config(dataset = Dataset.GOWALLA, domain = [(-122.446660, -122.382287), (37.747002, 37.809008)], experiment_series = 'gowalla/sf', true_radius_constant = 80, sparsity_constant = 1.136, early_stopping_levels = {1: 1, 3: 1, 5: 1, 7: 1}, epsilons = [2 + 1 * i for i in range(8)], laplace_epsilons = [15 + 25 * i for i in range(8)], sm_epsilons = [3.5 + 1.5 * i for i in range(8)], client_batch_size=500)
gowalla_austin_config = Config(dataset = Dataset.GOWALLA, domain = [(-97.755595,-97.717143), (30.266468, 30.294931)], experiment_series = 'gowalla/austin', true_radius_constant = 15000, sparsity_constant = 1.136)
gowalla_nyc_config = Config(dataset = Dataset.GOWALLA, domain = [(-74.2589, -73.7004), (40.4774, 40.9176)], experiment_series = 'gowalla')

one_dim_config_3 = Config(domain = [(0, math.sqrt(10 ** 7))], server_size = 10 ** 3, experiment_series = '1d', early_stopping_levels={1: 1, 3: 1.25, 5: 1.66, 9: 5}, true_radius_constant=1000, sparsity_constant = 2)
one_dim_config_4 = Config(domain = [(0, math.sqrt(10 ** 7))], server_size = 10 ** 4, experiment_series = '1d', early_stopping_levels={1: 1, 3: 1.17, 5: 1.39, 9: 2.32}, true_radius_constant=100, sparsity_constant = 1.44)
one_dim_config_5 = Config(domain = [(0, math.sqrt(10 ** 7))], server_size = 10 ** 5, experiment_series = '1d', early_stopping_levels={1: 1, 3: 1.14, 5: 1.3, 9: 1.9}, true_radius_constant=10, sparsity_constant = 1.17)
one_dim_config_6 = Config(domain = [(0, math.sqrt(10 ** 7))], server_size = 10 ** 6, experiment_series = '1d', early_stopping_levels={1: 1, 3: 1.14, 5: 1.3, 9: 1.9}, true_radius_constant=1, sparsity_constant = 1)
