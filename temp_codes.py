def Proximity(theta_cf, nbrs_gt, theta_gt):

    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1,-1))
    dist_cf_a0 = dist_cf_a0[0,0]
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]

    ## Calculating the proximity value
    distance = (dist_cf_a0 / dist_a0_xi)
    distance = 99999 if distance == 0 else distance
    return distance

import numpy as np
from sklearn.cluster import DBSCAN

def Connectedness(theta_cf, nbrs_gt, theta_gt):

    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1, -1))
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]
    epsilon =  dist_a0_xi + 0.001

    ## Clustering the potential counterfactual along with correctly classified instances
    theta_gt = theta_gt[ind_xi[0]]
    theta = np.r_[theta_gt, theta_cf.reshape(1, -1)]
    clustering = DBSCAN(eps=epsilon, min_samples=2, metric='minkowski', p=2).fit(theta)

    ## Calculating eps-chain
    if clustering.labels_[0] == clustering.labels_[-1]:
        chain = len(np.where(clustering.labels_ == clustering.labels_[0])[0])
    else:
        chain = 0

    return chain

import numpy as np
from sklearn.cluster import DBSCAN

def Connectedness(theta_cf, nbrs_gt, theta_gt, epsilon):

    ## Finding closet correctly predicted instances to counterfactual
    distances, indices = nbrs_gt.kneighbors(theta_cf.reshape(1, -1))
    theta_N = theta_gt[indices[0]]

    ## Clustering the counterfactual along with correctly classified instances
    theta_C = np.r_[theta_N, theta_cf.reshape(1, -1)]
    clustering = DBSCAN(eps=epsilon, min_samples=2, metric='minkowski', p=2).fit(theta_C)

    ## Measuring the epsilon-chain
    if clustering.labels_[-1] != -1:
        # chain = len(np.where(clustering.labels_[-1] == clustering.labels_)[0])
        chain = 1
    else:
        chain = 0

    return chain

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def Proximity(theta_cf, nbrs_gt, theta_gt):

    distances, indices = nbrs_gt.kneighbors(theta_cf.reshape(1, -1))
    theta_N = theta_gt[indices[0]]

    theta_O = np.r_[theta_N, theta_cf.reshape(1, -1)]

    lof_model = LocalOutlierFactor(n_neighbors=20)
    lof_model.fit_predict(theta_O)

    lof_score = lof_model.negative_outlier_factor_
    lof_score = (lof_score.max() - lof_score) / (lof_score.max() - lof_score.min())
    cf_score = lof_score[-1]

    return cf_score




import array
import numpy as np
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import hypervolume
from cost_function import CostFunction
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances

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
    similarity_vec = [0.1,0.3,0.6]

    ## Calculating epsilon
    dist = pairwise_distances(theta_gt, metric='minkowski')
    dist[np.where(dist==0)] = np.inf
    epsilon = np.min(dist)

    ## EA definition
    NDIM = len(x)
    BOUND_LOW, BOUND_UP = theta_min, theta_max
    OBJ_DIR = (-1.0, -1.0, -1.0, -1.0)    # -1.0: cost function | 1.0: fitness function
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=OBJ_DIR)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("evaluate", CostFunction, x, theta_x, discrete_indices, continuous_indices,
                     mapping_scale, mapping_offset, feature_range, blackbox, probability_range,
                     response_range, cf_label, nbrs_gt, theta_gt, epsilon)
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



import numpy as np
from gower_distance import GowerDistance
from prediction_distance import PredictionDistance
from proximity import Proximity
from connectedness import Connectedness
from sparsity import Sparsity

def CostFunction(x, theta_x, discrete_indices, continuous_indices,
                 mapping_scale, mapping_offset, feature_range, blackbox,
                 probability_range, response_range, cf_label,
                 nbrs_gt, theta_gt, epsilon, theta_cf):

    ## Constructing the counterfactual instance
    theta_cf = np.asarray(theta_cf)
    cf = theta_cf * mapping_scale + mapping_offset
    cf[discrete_indices] = np.rint(cf[discrete_indices])

    ## Objective 1: opposite outcome
    f1 = PredictionDistance(cf, blackbox, probability_range, response_range, cf_label)

    ## Objective 2: distnace
    f2 = GowerDistance(x, cf, feature_range, discrete_indices, continuous_indices)

    ## Objective 3: proximity
    f3 = Proximity(theta_cf, nbrs_gt, theta_gt)

    ## Objective 4: actionable
    # f4 = 0

    ## Objective 5: sparsity
    f5 = Sparsity(x, cf, feature_range, discrete_indices, continuous_indices, crisp_thresh=0.0)

    ## Objective 6: connectedness
    # f6 = Connectedness(theta_cf, nbrs_gt, theta_gt, epsilon)

    return f1, f2, f3, f5


def Proximity(theta_cf, nbrs_gt, theta_gt):

    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1,-1))
    dist_cf_a0 = dist_cf_a0[0,0]
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]

    ## Calculating the proximity value
    distance = (dist_cf_a0 / dist_a0_xi)
    distance = 99999 if distance == 0 else distance
    return distance


import numpy as np

def Sparsity(x1, x2, feature_range, discrete_indices, continuous_indices, crisp_thresh=0.1):
    changed = []
    if continuous_indices is not None:
        for j in continuous_indices:
            changed.append((1/feature_range[j]) * abs(x1[j]-x2[j]))

    if discrete_indices is not None:
        for j in discrete_indices:
            changed.append(int(x1[j] != x2[j]))

    changed = np.asarray(changed)
    changed[np.where(changed > crisp_thresh)] = 1
    changed[np.where(changed < crisp_thresh)] = 0
    return sum(changed)


import numpy as np
from sklearn.cluster import DBSCAN

def Connectedness(theta_cf, nbrs_gt, theta_gt, epsilon):
    ## Finding closet correctly predicted instances to counterfactual
    distances, indices = nbrs_gt.kneighbors(theta_cf.reshape(1, -1))
    theta_N = theta_gt[indices[0]]

    ## Clustering the counterfactual along with correctly classified instances
    theta_C = np.r_[theta_N, theta_cf.reshape(1, -1)]
    clustering = DBSCAN(eps=epsilon, min_samples=2, metric='minkowski', p=2).fit(theta_C)

    ## Measuring the epsilon-chain
    chain = 0 if clustering.labels_[-1] == -1 else 1
    return chain

def Sparsity(theta_x, theta_cf):
    return sum(theta_x != theta_cf)


import numpy as np
def Proximity(theta_cf, nbrs_gt, theta_gt):
    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1,-1))
    dist_cf_a0 = dist_cf_a0[0,0]
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]

    ## Calculating the proximity value
    distance = (dist_cf_a0 / dist_a0_xi)
    return distance
