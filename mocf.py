import random, array
import numpy as np
from deap import algorithms, base, creator, tools
from cost_function import CostFunction

def Uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def RecoverCounterfactuals(fronts,mapping_scale, mapping_offset, discrete_indices):
    P = np.asarray(fronts[0])
    cf_set = P * mapping_scale + mapping_offset
    cf_set[:,discrete_indices] = np.rint(cf_set[:,discrete_indices])
    return cf_set

def MOCF(x, blackbox, dataset, probability_range):

    l_x = blackbox.predict(x.reshape(1,-1))
    l_cf = int(1 - l_x)     # Desired label

    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = feature_max - feature_min
    theta_min = -1
    theta_max = 1

    mapping_scale = (feature_max - feature_min) / (theta_max - theta_min)
    mapping_offset = -theta_min * (feature_max - feature_min) / (theta_max - theta_min) + feature_min

    NDIM = len(x)
    BOUND_LOW, BOUND_UP = theta_min, theta_max
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("evaluate", CostFunction, x, l_cf, discrete_indices, continuous_indices,
                     mapping_scale, mapping_offset, feature_range, blackbox, probability_range)
    toolbox.register("attr_float", Uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    toolbox.register("select", tools.selNSGA2)

    toolbox.pop_size = 100
    toolbox.max_gen = 200
    toolbox.mut_prob = 0.2

    def run_ea(toolbox, stats=None, verbose=False):
        pop = toolbox.population(n=toolbox.pop_size)
        pop = toolbox.select(pop, len(pop))
        return algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.pop_size,
                                         lambda_=toolbox.pop_size,
                                         cxpb=1 - toolbox.mut_prob,
                                         mutpb=toolbox.mut_prob,
                                         stats=stats,
                                         ngen=toolbox.max_gen,
                                         verbose=verbose)

    res, _ = run_ea(toolbox)
    fronts = tools.emo.sortLogNondominated(res, len(res))
    cf_set = RecoverCounterfactuals(fronts, mapping_scale, mapping_offset, discrete_indices)
    cf_set_y = blackbox.predict_proba(cf_set)

    print('end')

