import math
from schedulers import *
from data import Dataset

class Config:
    def __init__(
        self,
        domain = [(0, math.sqrt(10 ** 7)), (0, math.sqrt(10 ** 7))],
        server_size = 10 ** 5,
        client_batch_size = 500,
        early_stopping_levels = [1, 3, 5, 9],
        epsilons = [0.01, 0.1, 1, 1.5, 2, 3, 4, 5, 7, 10, 12, 17],
        scheduler_types = [constant],
        dir_prefix = 'graphs/densities',
        dataset = Dataset.RANDOM
    ):
        self.domain = domain
        self.server_size = server_size
        self.client_batch_size = client_batch_size
        self.early_stopping_levels = early_stopping_levels
        self.epsilons = epsilons
        self.scheduler_types = scheduler_types
        self.dir_prefix = dir_prefix

    @property
    def client_sensitivity(self):
        return math.sqrt(sum([(upper - lower) ** 2 for (lower, upper) in self.domain]))
    
    @property
    def server_area(self):
        return math.prod([upper - lower for (lower, upper) in self.domain])

    @property
    def true_radius_squared(self):
        return self.server_area / self.server_size * 10
    
    @property
    def scheduler_type_to_schedulers(self):
        return {scheduler_type.__name__: [scheduler_type(eps) for eps in self.epsilons] for scheduler_type in self.scheduler_types}
    
    @property
    def output_dir(self):
        return f'{self.dir_prefix}/size_{self.server_size}_domain_{int(self.server_area)}'

# SCHEDULER_TYPES = [constant, linear, log, sqrt, quadratic, reverse_linear, inverse_sqrt, inverse_linear, inverse_quadratic]

density_config_3 = Config(server_size = 10 ** 3)
density_config_4 = Config(server_size = 10 ** 4)
density_config_5 = Config(server_size = 10 ** 5)
density_config_6 = Config(server_size = 10 ** 6)

gowalla_sf_config = Config(server_size = 10 ** 5, dataset = Dataset.GOWALLA, domain = [(37.5395, 37.7910), (-122.5153, -122.3789)])
gowalla_austin_config = Config(server_size = 10 ** 5, dataset = Dataset.GOWALLA, domain = [(29.833, 30.838), (-96.700, -98.592)])
gowalla_nyc_config = Config(server_size = 10 ** 5, dataset = Dataset.GOWALLA, domain = [(40.4774, 40.9176), (-74.2589, -73.7004)])

