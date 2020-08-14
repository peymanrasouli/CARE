import array
import numpy as np
import pandas as pd
from math import *
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import convergence, diversity, hypervolume, igd
from pymoo.factory import get_performance_indicator
from cost_function import CostFunction, FeatureDistance
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import pairwise_distances
import hdbscan

def Initialization(bound_low, bound_up, size, theta_x, theta_N, similarity_vec):
    method = np.random.choice(['x','neighbor','random'], size=1, replace=False, p=similarity_vec)
    if method == 'x':
        return list(theta_x)
    elif method == 'neighbor':
        idx = np.random.choice(range(len(theta_N)), size=1, replace=False)
        return list(theta_N[idx].ravel())
    elif method == 'random':
        return list(np.random.uniform(bound_low, bound_up, size))

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

def SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x, theta_x, discrete_indices, continuous_indices,
                 mapping_scale, mapping_offset, feature_range, blackbox, probability_range, response_range,
                 cf_label, theta_N, similarity_vec, lof_model, hdbscan_model, actions_o, actions_w):
    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=OBJ_W)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register("evaluate", CostFunction, x, discrete_indices, continuous_indices,
                     mapping_scale, mapping_offset, feature_range, blackbox, probability_range,
                     response_range, cf_label, lof_model, hdbscan_model,  actions_o, actions_w)
    toolbox.register("attr_float", Initialization, BOUND_LOW, BOUND_UP, NDIM, theta_x, theta_N, similarity_vec)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    # toolbox.register("mate", tools.cxUniform, indpb=1.0 / NDIM)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    ref_points = tools.uniform_reference_points(NOBJ, P)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox

def RunEA(toolbox, MU, NGEN, CXPB, MUTPB):
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(pop=pop, gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(pop=pop, gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    fronts = tools.emo.sortLogNondominated(pop, MU)
    return fronts, pop, record, logbook

def ConstructCounterfactuals(toolbox, fronts, dataset, mapping_scale, mapping_offset, blackbox, cf_label, priority):

    ## Constructing counterfactuals
    pop = []
    evaluation = []
    for f in fronts:
        for ind in f:
            pop.append(np.asarray(ind))
            evaluation.append(np.asarray(toolbox.evaluate(ind)))
    pop = np.asarray(pop)
    evaluation = np.asarray(evaluation)
    solutions = (pop * mapping_scale + mapping_offset).astype(int)

    cfs = pd.DataFrame(data=solutions, columns=dataset['feature_names'])
    label = blackbox.predict(cfs)
    evaluation = np.c_[evaluation,label]
    cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity',
                                                      'Actionable', 'Sparsity', 'Connectedness', 'Label'])

    ## Applying compulsory conditions
    drop_indices = cfs_eval[(cfs_eval['Prediction'] > 0.1) | (cfs_eval['Proximity'] == -1) |
                            (cfs_eval['Connectedness'] == 0) |  (cfs_eval['Label'] != cf_label)].index
    cfs.drop(drop_indices, inplace=True)
    cfs_eval.drop(drop_indices, inplace=True)

    ## Sorting counterfactuals based on priority
    sort_indices = cfs_eval.sort_values(by=list(priority.keys()), ascending=list(priority.values())).index
    cfs = cfs.reindex(sort_indices)
    cfs_eval = cfs_eval.reindex(sort_indices)

    ## Dropping duplicate counterfactuals
    cfs = cfs.drop_duplicates()
    cfs_eval = cfs_eval.reindex(cfs.index)

    ## Predicting counterfactuals
    if cf_label is None:
        cfs_prob = None
        cfs_y = blackbox.predict(cfs)
        return cfs, cfs_y, cfs_prob, cfs_eval
    else:
        cfs_y = blackbox.predict(cfs)
        cfs_prob = blackbox.predict_proba(cfs)
        return cfs, cfs_y, cfs_prob, cfs_eval

def MOCF(x, blackbox, dataset, X_train, Y_train, probability_range=None, response_range=None, cf_label=None):

    ## Reading dataset information
    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = feature_max - feature_min

    ## Linear mapping
    theta_min = -1
    theta_max = 1
    mapping_scale = (feature_max - feature_min) / (theta_max - theta_min)
    mapping_offset = -theta_min * (feature_max - feature_min) / (theta_max - theta_min) + feature_min
    theta_x = (x - mapping_offset) / mapping_scale

    ## KNN model of correctly classified samples same class as counterfactual
    pred_train = blackbox.predict(X_train)
    gt = X_train[np.where(pred_train == Y_train)]
    pred_gt = blackbox.predict(gt)
    gt = gt[np.where(pred_gt == cf_label)]
    theta_gt = (gt - mapping_offset) / mapping_scale
    nbrs_gt = NearestNeighbors(n_neighbors=min(len(gt),200), algorithm='kd_tree').fit(theta_gt)

    ## Initialization
    distances, indices = nbrs_gt.kneighbors(theta_x.reshape(1,-1))
    theta_N = theta_gt[indices[0]].copy()
    similarity_vec = [0.1,0.2,0.7]

    ## Creating local outlier factor model
    lof_model = LocalOutlierFactor(n_neighbors=1, novelty=True)
    lof_model.fit(theta_gt)

    ## Creating hdbscan clustering model
    dist = pairwise_distances(theta_gt, metric='minkowski')
    dist[np.where(dist==0)] = np.inf
    epsilon = np.max(np.min(dist,axis=0))
    hdbscan_model = hdbscan.HDBSCAN(min_samples=2, cluster_selection_epsilon=float(epsilon),
                                    metric='minkowski', p=2, prediction_data=True).fit(theta_gt)

    ## Actionable operation vector
    if dataset['name'] == 'breast-cancer':
        desired_actions = {
            'age': (np.greater_equal, 5),
            'tumor-size': (np.greater_equal,8),
            'inv-node': (np.less_equal, 3),
            'breast': (np.equal, 1),
        }
        actions_o = [0] * len(x)
        actions_w = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions_o[index] = desired_actions[a][0]
            actions_w[index] = desired_actions[a][1]


    elif dataset['name'] == 'credit-card-default':
        desired_actions = {
            'SEX': (np.equal, 10),
            'EDUCATION': (np.greater_equal, 3),
            'MARRIAGE': (np.equal, 8),
            'AGE': (np.greater_equal, 5)
        }
        actions_o = [0] * len(x)
        actions_w = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions_o[index] = desired_actions[a][0]
            actions_w[index] = desired_actions[a][1]


    elif dataset['name'] == 'adult':
        desired_actions = {
            'education': (np.greater_equal, 5),
            'marital-status': (np.equal, 8),
            'race': (np.equal, 10),
            'sex': (np.equal, 10),
            'native-country': (np.equal, 7),
            'age': (np.greater_equal, 5),
            'capital-gain': (np.equal, 10)
        }
        actions_o = [0] * len(x)
        actions_w = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions_o[index] = desired_actions[a][0]
            actions_w[index] = desired_actions[a][1]

    ## n-Closest ground truth counterfactual in the training data
    n = 5
    dist = np.asarray([FeatureDistance(x, cf_, feature_range, discrete_indices, continuous_indices) for cf_ in gt])
    closest_ind = np.argsort(dist)[:n]
    for i in range(n):
        theta_cf = (gt[closest_ind[i]] - mapping_offset) / mapping_scale
        print('instnace:', list(gt[closest_ind[i]]), 'probability:',
              blackbox.predict_proba(gt[closest_ind[i]].reshape(1,-1)),
              'cost:',CostFunction(x, discrete_indices, continuous_indices,
              mapping_scale, mapping_offset, feature_range, blackbox,
              probability_range, response_range, cf_label, lof_model,
              hdbscan_model, actions_o, actions_w, theta_cf))

    ## Parameter setting
    NDIM = len(x)
    NOBJ = 6
    NGEN = 100
    CXPB = 0.5
    MUTPB = 0.2
    P = 8
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    BOUND_LOW, BOUND_UP = theta_min, theta_max

    ##  Objective functions || -1.0: cost function | 1.0: fitness function
    f1 = -1.0   # Prediction Distance
    f2 = -1.0   # Feature Distance
    f3 =  1.0   # Proximity
    f4 = -1.0   # Actionable Recourse
    f5 = -1.0   # Sparsity
    f6 =  1.0   # Connectedness
    OBJ_W = (f1, f2, f3, f4, f5, f6)

    ## Creating toolbox
    toolbox = SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x, theta_x, discrete_indices, continuous_indices,
                           mapping_scale, mapping_offset, feature_range, blackbox, probability_range, response_range,
                           cf_label, theta_N, similarity_vec, lof_model, hdbscan_model, actions_o, actions_w)

    ## Running EA
    fronts, pop, record, logbook= RunEA(toolbox, MU, NGEN, CXPB, MUTPB)

    print('done')


    ################## Decision making techniques ######################

    ## Decision making using user-defined priority
    priority = {
        'Distance': 1,
        'Sparsity': 1,
        'Actionable': 1,
        'Connectedness': 0,
        'Proximity': 0,
        'Prediction': 1,
        'Label': 1
        }
    cfs, cfs_y, cfs_prob, cfs_eval = ConstructCounterfactuals(toolbox, fronts, dataset,
                                                              mapping_scale, mapping_offset,
                                                              blackbox, cf_label, priority)

    ## Decision making using Pseudo-Weights



    ## Decision making using High Trade-Off Solutions






    ############### Benchmarks and Evaluations ####################

    ## Calculating Hypervolume
    pops = logbook.select('pop')
    pops_obj = [np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
    hypervols = [hypervolume(pop, ref) for pop in pops]
    plt.plot(hypervols)
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')


    ## Calculating GD, IGD



    ## Calculating Convergence
