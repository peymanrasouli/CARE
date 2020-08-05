import array
import numpy as np
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import hypervolume
import matplotlib.pyplot as plt
from cost_function import CostFunction

def CalculateReferencePoint(toolbox, fronts):
    fronts = np.concatenate(fronts)
    obj_vals = [toolbox.evaluate(ind) for ind in fronts]
    reference_point = np.max(obj_vals, axis=0)
    return reference_point

def Uniform(bound_low, bound_up, size):
    return list(np.random.uniform(bound_low, bound_up, size))

def RecoverCounterfactuals(fronts,mapping_scale, mapping_offset, discrete_indices):
    P = np.concatenate(fronts)
    CFs = P * mapping_scale + mapping_offset
    CFs[:,discrete_indices] = np.rint(CFs[:,discrete_indices])
    return CFs

def PlotParetoFronts(toolbox, fronts, objective_list):
    n_fronts = len(fronts)
    fig, ax = plt.subplots(n_fronts, figsize=(8,8))
    fig.text(0.5, 0.04, 'f' + str(objective_list[0] + 1) + '(x)', ha='center')
    fig.text(0.02, 0.5, 'f' + str(objective_list[1] + 1) + '(x)', rotation='vertical')
    for i, f in enumerate(fronts):
        costs = np.asarray([toolbox.evaluate(ind) for ind in f])[:,objective_list]
        ax.scatter(costs[:,0], costs[:,1], color='r') if n_fronts == 1 \
            else ax[i].scatter(costs[:,0], costs[:,1], color='r')
        ax.title.set_text('Front 1') if n_fronts == 1 else ax[i].title.set_text('Front '+str(i+1))

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

    toolbox.pop_size = 200
    toolbox.max_gen = 200
    toolbox.mut_prob = 0.4

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



    results, logbook = run_ea(toolbox)
    fronts = tools.emo.sortLogNondominated(results, len(results))
    PlotParetoFronts(toolbox, fronts, objective_list=[0, 1])

    CFs = RecoverCounterfactuals(fronts, mapping_scale, mapping_offset, discrete_indices)
    CFs_y = blackbox.predict_proba(CFs)

    reference_point = CalculateReferencePoint(toolbox, fronts)

    hyper_volume = hypervolume(results,reference_point)

    print('done')

