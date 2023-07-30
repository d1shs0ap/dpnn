import math

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
    height = 14
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