import math

HEIGHT = 12

def constant(eps):
    return lambda level: eps

def linear_decrease_1(eps):
    return lambda level: 1.3 * eps - 0.05 * eps * level

def linear_decrease_2(eps):
    return lambda level: 1.6 * eps - 0.1 * eps * level

def linear_decrease_3(eps):
    return lambda level: max(2.2 * eps - 0.207 * eps * level, 0.1)

def linear_decrease_4(eps):
    return lambda level: max(3.4 * eps - 0.543 * eps * level, 0.1)


# show increasing epsilon is bad, when pronounced
def linear_increase_1(eps):
    return lambda level: linear_decrease_1(eps)(HEIGHT - level)

def linear_increase_2(eps):
    return lambda level: linear_decrease_2(eps)(HEIGHT - level)

def linear_increase_3(eps):
    return lambda level: linear_decrease_3(eps)(HEIGHT - level)

def linear_increase_4(eps):
    return lambda level: linear_decrease_4(eps)(HEIGHT - level)


# def log(eps):
#     return lambda level: 1.5 * eps - 0.28 * eps * math.log(level + 1)

# def sqrt(eps):
#     return lambda level: 1.5 * eps - 0.2 * eps * math.sqrt(level + 1)

# def quadratic(eps):
#     return lambda level: 1.5 * eps - 0.008 * eps * (level + 1) ** 2

# # show decreasing too fast is bad
# def inverse_sqrt(eps):
#     return lambda level: 2.15 * eps / (level + 1) ** (1 / 2)

# def inverse_linear(eps):
#     return lambda level: 4 * eps / (level + 1)

# def inverse_quadratic(eps):
#     return lambda level: 8 * eps / (level + 1) ** 2

# def best_scheduler(eps):
#     return lambda level: 1.3 * eps - 0.17 * eps * math.sqrt(level + 1)