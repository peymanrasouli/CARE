import array
import numpy as np
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import hypervolume
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from cost_function import CostFunction

def CalculateReferencePoint(toolbox, fronts):
    fronts = np.concatenate(fronts)
    obj_vals = [toolbox.evaluate(ind) for ind in fronts]
    reference_point = np.max(obj_vals, axis=0)
    return reference_point

def Initialization(bound_low, bound_up, size, theta_x, theta_N, similarity_vec):
    method = np.random.choice(['x','neighbor','random'], size=1, replace=False, p=similarity_vec)
    if method == 'x':
        return list(theta_x)
    elif method == 'neighbor':
        idx = np.random.choice(range(len(theta_N)), size=1, replace=False)
        return list(theta_N[idx].ravel())
    elif method == 'random':
        return list(np.random.uniform(bound_low, bound_up, size))

def ConstructCounterfactuals(fronts,mapping_scale, mapping_offset, discrete_indices):
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

def MOCF(x, blackbox, dataset, X_train, Y_train, probability_range=None, response_range=None, cf_label=None):

    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = feature_max - feature_min
    theta_min = -1
    theta_max = 1

    ## Linear mapping
    mapping_scale = (feature_max - feature_min) / (theta_max - theta_min)
    mapping_offset = -theta_min * (feature_max - feature_min) / (theta_max - theta_min) + feature_min
    theta_x = (x - mapping_offset) / mapping_scale

    ## Initialization
    pred_train = blackbox.predict(X_train)
    N = X_train[np.where(pred_train==cf_label)]
    theta_N = (N - mapping_offset) / mapping_scale
    nbrs = NearestNeighbors(n_neighbors=min(len(N),100), algorithm='kd_tree').fit(theta_N)
    distances, indices = nbrs.kneighbors(theta_x.reshape(1,-1))
    theta_N = theta_N[indices[0]]
    similarity_vec = [0.1,0.3,0.6]

    ## KNN model of correctly classified samples for counterfactual
    gt = X_train[np.where(pred_train == Y_train)]
    pred_gt = blackbox.predict(gt)
    gt = gt[np.where(pred_gt == cf_label)]
    theta_gt = (gt - mapping_offset) / mapping_scale
    nbrs_gt = NearestNeighbors(n_neighbors=min(len(gt),300), algorithm='kd_tree').fit(theta_gt)

    ## EA definition
    NDIM = len(x)
    BOUND_LOW, BOUND_UP = theta_min, theta_max
    OBJ_DIR = (-1.0, -1.0, -1.0, -1.0, 1.0)    # -1.0: cost function | 1.0: fitness function
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=OBJ_DIR)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("evaluate", CostFunction, x, theta_x, discrete_indices, continuous_indices,
                     mapping_scale, mapping_offset, feature_range, blackbox, probability_range,
                     response_range, cf_label, nbrs_gt, theta_gt)
    toolbox.register("attr_float", Initialization, BOUND_LOW, BOUND_UP, NDIM, theta_x, theta_N, similarity_vec)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=1.0 / NDIM)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    toolbox.register("select", tools.selNSGA2)

    toolbox.pop_size = 100
    toolbox.max_gen = 100
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



    results, logbook = run_ea(toolbox)
    fronts = tools.emo.sortLogNondominated(results, len(results))
    PlotParetoFronts(toolbox, fronts, objective_list=[0, 1])

    CFs = ConstructCounterfactuals(fronts, mapping_scale, mapping_offset, discrete_indices)
    if cf_label is None:
        CFs_y = blackbox.predict(CFs)
    else:
        CFs_y = blackbox.predict(CFs)
        CFs_prob = blackbox.predict_proba(CFs)

    reference_point = CalculateReferencePoint(toolbox, fronts)
    hyper_volume = hypervolume(results,reference_point)

    print('done')

