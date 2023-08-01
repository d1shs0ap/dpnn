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
        experiment_series = 'densities',
        dataset = Dataset.RANDOM,
        true_radius_constant = 10
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
    def scheduler_type_to_schedulers(self):
        return {scheduler_type.__name__: [scheduler_type(eps) for eps in self.epsilons] for scheduler_type in self.scheduler_types}
    
    @property
    def output_dir(self):
        return f'graphs/{self.experiment_series}/size_{self.server_size}_domain_{int(self.server_area)}'

# SCHEDULER_TYPES = [constant, linear, log, sqrt, quadratic, reverse_linear, inverse_sqrt, inverse_linear, inverse_quadratic]

density_config_3 = Config(server_size = 10 ** 3)
density_config_4 = Config(server_size = 10 ** 4)
density_config_5 = Config(server_size = 10 ** 5)
density_config_6 = Config(server_size = 10 ** 6)

gowalla_sf_config = Config(dataset = Dataset.GOWALLA, domain = [(-122.443021, -122.399762), (37.769949, 37.803729)], experiment_series = 'gowalla', true_radius_constant = 20000)
gowalla_austin_config = Config(dataset = Dataset.GOWALLA, domain = [(-96.700, -98.592), (29.833, 30.838)], experiment_series = 'gowalla')
gowalla_nyc_config = Config(dataset = Dataset.GOWALLA, domain = [(-74.2589, -73.7004), (40.4774, 40.9176)], experiment_series = 'gowalla')

leaf_config_3 = Config(server_size = 10 ** 3, experiment_series = 'leaf', early_stopping_levels=[3, 5, 9])
leaf_config_4 = Config(server_size = 10 ** 4, experiment_series = 'leaf', early_stopping_levels=[3, 5, 9])
leaf_config_5 = Config(server_size = 10 ** 5, experiment_series = 'leaf', early_stopping_levels=[3, 5, 9])