import array
import numpy as np
import pandas as pd
from math import *
from deap import algorithms, base, creator, tools
from mappings import ord2ohe, ord2org, ord2theta, theta2ord, theta2org, org2ord
from cost_function import CostFunction
from evaluate_counterfactuals import evaluateCounterfactuals
# from recover_originals import RecoverOriginals
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import pairwise_distances, f1_score, r2_score
from dython import nominal
import hdbscan
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

def Initialization(bound_low, bound_up, size, x_theta, nbrs_theta, selection_probability):
    method = np.random.choice(['x','neighbor','random'], size=1, p=list(selection_probability.values()))
    if method == 'x':
        return list(x_theta)
    elif method == 'neighbor':
        idx = np.random.choice(range(len(nbrs_theta)), size=1)
        return list(nbrs_theta[idx].ravel())
    elif method == 'random':
        return list(np.random.uniform(bound_low, bound_up, size))

def SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x_ord, x_theta, x_org, dataset, predict_class_fn,
                 predict_proba_fn, discrete_indices, continuous_indices, feature_width, ea_scaler,
                 probability_thresh, cf_class, cf_range, nbrs_theta, selection_probability,
                 lof_model, hdbscan_model, action_operation, action_priority, corr_models):

    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=OBJ_W)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register("evaluate", CostFunction, x_ord, x_theta, x_org, dataset, predict_class_fn, predict_proba_fn,
                     discrete_indices, continuous_indices, feature_width, ea_scaler, probability_thresh,
                     cf_class, cf_range, lof_model, hdbscan_model, action_operation, action_priority, corr_models)

    toolbox.register("attr_float", Initialization, BOUND_LOW, BOUND_UP, NDIM, x_theta, nbrs_theta, selection_probability)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    ref_points = tools.uniform_reference_points(NOBJ, P)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    # toolbox.register("select", tools.selNSGA2)
    return toolbox

def RunEA(toolbox, MU, NGEN, CXPB, MUTPB, SNAPSHOT_STEP):
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
    snapshot = []
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

        if not((gen+1) % SNAPSHOT_STEP):
            snapshot.append(tools.emo.sortLogNondominated(pop, MU)[0])
    snapshot.append(tools.emo.sortLogNondominated(pop, MU)[0])

    fronts = tools.emo.sortLogNondominated(pop, MU)
    return fronts, pop, snapshot, record, logbook

def MOCFExplainer(x_ord, blackbox, predict_class_fn, predict_proba_fn, dataset, task, X_train, Y_train,
                  probability_thresh=None, cf_class=None, x_range=None, cf_range=None, preferences=None):

    # Scaling data to (0,1) for EA
    ea_scaler = MinMaxScaler(feature_range=(0,1))
    ea_scaler.fit(X_train)
    X_train_theta = ord2theta(X_train, ea_scaler)

    ## KNN model of correctly classified samples same class as counter-factual
    X_train_ohe = ord2ohe(X_train, dataset)
    pred_train = predict_class_fn(X_train_ohe)
    if cf_class is None:
        abs_error = np.abs(Y_train-pred_train)
        mae = np.mean(abs_error)
        gt_idx = np.where(abs_error<=mae)
        pred_gt = predict_class_fn(X_train_ohe[gt_idx])
        gt_sc_idx = np.where(np.logical_and(pred_gt >= cf_range[0], pred_gt <= cf_range[1]))
        gt_ord = X_train[gt_idx[0][gt_sc_idx[0]]]
        gt_theta = ord2theta(gt_ord, ea_scaler)
    else:
        gt_idx = np.where(pred_train == Y_train)
        pred_gt = predict_class_fn(X_train_ohe[gt_idx])
        gt_sc_idx = np.where(pred_gt == cf_class)
        gt_ord = X_train[gt_idx[0][gt_sc_idx[0]]]
        gt_theta = ord2theta(gt_ord, ea_scaler)

    ## Creating local outlier factor model for proximity function
    lof_model = LocalOutlierFactor(n_neighbors=1, novelty=True)
    lof_model.fit(gt_theta)

    ## Creating hdbscan clustering model for connectedness function
    dist = pairwise_distances(gt_theta, metric='minkowski')
    dist[np.where(dist==0)] = np.inf
    epsilon = np.max(np.min(dist,axis=0))
    hdbscan_model = hdbscan.HDBSCAN(min_samples=2, cluster_selection_epsilon=float(epsilon),
                                    metric='minkowski', p=2, prediction_data=True).fit(gt_theta)

    ## Feature correlation modeling
    # Calculate the correlation/strength-of-association of features in data-set
    # with both categorical and continuous features using:
    # Pearson's R for continuous-continuous cases
    # Correlation Ratio for categorical-continuous cases
    # Cramer's V for categorical-categorical cases
    continuous_indices = dataset['continuous_indices']
    discrete_indices = dataset['discrete_indices']
    corr = nominal.associations(X_train, nominal_columns=discrete_indices,theil_u=False, plot=False)['corr']
    corr = corr.to_numpy()
    corr[np.diag_indices(corr.shape[0])] = 0
    corr_thresh = 0.2
    corr_features = np.where(abs(corr) > corr_thresh)
    corr_ = np.zeros(corr.shape)
    corr_[corr_features] = 1

    ## Correlation models
    val_idx = int(0.7 * len(X_train))
    corr_models = []
    for f in range(len(corr_)):
        inputs = np.where(corr_[f,:] == 1)[0]
        if len(inputs) > 0:
            if f in discrete_indices:
                model = DecisionTreeClassifier()
                model.fit(X_train_theta[0:val_idx, inputs], X_train[0:val_idx, f])
                score = f1_score(X_train[val_idx:, f], model.predict(X_train_theta[val_idx:, inputs]), average='micro')
                if score > 0.7:
                    corr_models.append({'feature': f, 'inputs': inputs, 'model': model, 'score': score})
            elif f in continuous_indices:
                model = DecisionTreeRegressor()
                model.fit(X_train_theta[0:val_idx,inputs], X_train[0:val_idx,f])
                score = r2_score(X_train[val_idx:,f], model.predict(X_train_theta[val_idx:,inputs]))
                if score > 0.7:
                    corr_models.append({'feature':f, 'inputs':inputs, 'model':model, 'score':score})


    ## Evolutioanry algorithm setup
    # Initializing the population
    K_nbrs = min(500, len(gt_theta))
    gt_nbrModel = NearestNeighbors(n_neighbors=K_nbrs, algorithm='kd_tree').fit(gt_theta)
    x_theta = ord2theta(x_ord, ea_scaler)
    distances, indices = gt_nbrModel.kneighbors(x_theta.reshape(1, -1))
    nbrs_theta = gt_theta[indices[0]].copy()
    selection_probability = {'x': 0.1, 'neighbor':0.4, 'random':0.5}

    # Objective functions || -1.0: cost function | 1.0: fitness function
    f1 = -1.0   # Prediction Distance
    f2 = -1.0   # Feature Distance
    f3 = -1.0   # Sparsity
    f4 =  1.0   # Proximity
    f5 =  1.0   # Connectedness
    f6 = -1.0   # Actionable Recourse
    f7 = -1.0   # Correlation


    OBJ_W = (f1, f2, f3, f4, f5, f6, f7)
    OBJ_name = ['Prediction', 'Distance', 'Sparsity', 'Proximity',  'Connectedness', 'Actionable', 'Correlation']

    # EA parameters
    NDIM = len(x_ord)
    NOBJ = len(OBJ_W)
    NGEN = 10
    CXPB = 0.6
    MUTPB = 0.2
    P = 6
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    SNAPSHOT_STEP = 3
    BOUND_LOW, BOUND_UP = 0, 1
    feature_width = np.max(X_train, axis=0) - np.min(X_train, axis=0)

    # Creating toolbox for the EA
    x_org = ord2org(x_ord, dataset)
    action_operation =  preferences['action_operation']
    action_priority = preferences['action_priority']
    toolbox = SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x_ord, x_theta, x_org, dataset, predict_class_fn,
                           predict_proba_fn, discrete_indices, continuous_indices, feature_width, ea_scaler,
                           probability_thresh, cf_class, cf_range, nbrs_theta, selection_probability, lof_model,
                           hdbscan_model, action_operation, action_priority, corr_models)

    ## Running EA
    fronts, pop, snapshot, record, logbook= RunEA(toolbox, MU, NGEN, CXPB, MUTPB, SNAPSHOT_STEP)

    ## Constructing counter-factuals
    # cfs_theta = np.asarray(fronts[0])
    cfs_theta = np.concatenate(np.asarray([s for s in snapshot]))

    feature_names = dataset['feature_names']
    cf_org = theta2org(cfs_theta, ea_scaler, dataset)
    cf_ord = org2ord(cf_org, dataset)
    cfs_ord = pd.DataFrame(data=cf_ord, columns=feature_names)

    ## Evaluating counter-factuals
    OBJ_order = [1, 2, 3, 4, 5, 6, 7]
    MOCF_output = {'toolbox': toolbox,
                   'ea_scaler': ea_scaler,
                   'OBJ_name': OBJ_name,
                   'OBJ_order': OBJ_order,
                   'OBJ_W': OBJ_W}

    cfs_ord, cfs_eval = EvaluateCounterfactuals(cfs_ord, dataset, predict_class_fn, predict_proba_fn, task, MOCF_output)

    ## Recovering original data
    x_org, cfs_org, x_cfs_org, x_cfs_highlight = RecoverOriginals(x_ord, cfs_ord, dataset)

    ## Returning the results
    output = {'cfs_ord': cfs_ord,
              'cfs_org': cfs_org,
              'cfs_eval': cfs_eval,
              'x_cfs_org': x_cfs_org,
              'x_cfs_highlight': x_cfs_highlight,
              'toolbox': toolbox,
              'ea_scaler': ea_scaler,
              'OBJ_name': OBJ_name,
              'OBJ_order': OBJ_order,
              'OBJ_W': OBJ_W
    }

    return output

