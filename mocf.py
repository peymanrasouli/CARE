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
from sklearn.metrics import pairwise_distances, mean_absolute_error
from dython import nominal
import hdbscan

def Initialization(bound_low, bound_up, size, theta_x, theta_N, probability_vec):
    method = np.random.choice(['x','neighbor','random'], size=1, replace=False, p=probability_vec)
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
                 mapping_scale, mapping_offset, feature_range, blackbox, probability_thresh, cf_label,
                 cf_range, theta_N, probability_vec, lof_model, hdbscan_model, actions, corr):
    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=OBJ_W)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register("evaluate", CostFunction, x, theta_x, discrete_indices, continuous_indices,
                     mapping_scale, mapping_offset, feature_range, blackbox, probability_thresh,
                     cf_label, cf_range, lof_model, hdbscan_model,  actions, corr)
    toolbox.register("attr_float", Initialization, BOUND_LOW, BOUND_UP, NDIM, theta_x, theta_N, probability_vec)
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

def FeatureDecoder(df, discrete_features, feature_encoder):
    df_de = df.copy(deep=True)
    for f in discrete_features:
        fe = feature_encoder[f]
        decoded_data = fe.inverse_transform(df_de[f].to_numpy().reshape(-1, 1))
        df_de[f] = decoded_data
    return df_de

def ConstructCounterfactuals(x, toolbox, fronts, dataset, mapping_scale, mapping_offset, blackbox, cf_label, priority):

    ## Constructing counterfactuals
    pop = []
    evaluation = []
    for f in fronts:
        for ind in f:
            pop.append(np.asarray(ind))
            evaluation.append(np.asarray(toolbox.evaluate(ind)))
    pop = np.asarray(pop)
    evaluation = np.asarray(evaluation)
    solutions = np.rint(pop * mapping_scale + mapping_offset)

    cfs = pd.DataFrame(data=solutions, columns=dataset['feature_names']).astype(int)
    if cf_label is None:
        response = blackbox.predict(cfs)
        evaluation = np.c_[evaluation, response]
        cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity', 'Actionable',
                                                          'Sparsity', 'Connectedness', 'Response'])
    else:
        label = blackbox.predict(cfs)
        prob = blackbox.predict_proba(cfs)[:,cf_label]
        evaluation = np.c_[evaluation, label, prob]
        cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity', 'Actionable',
                                                          'Sparsity', 'Connectedness', 'Label', 'Probability'])

    ## Applying compulsory conditions
    drop_indices = cfs_eval[(cfs_eval['Prediction'] > 0) | (cfs_eval['Proximity'] == -1) |
                            (cfs_eval['Connectedness'] == 0)].index
    cfs.drop(drop_indices, inplace=True)
    cfs_eval.drop(drop_indices, inplace=True)

    ## Sorting counterfactuals based on priority
    sort_indices = cfs_eval.sort_values(by=list(priority.keys()), ascending=list(priority.values())).index
    cfs = cfs.reindex(sort_indices)
    cfs_eval = cfs_eval.reindex(sort_indices)

    ## Dropping duplicate counterfactuals
    cfs = cfs.drop_duplicates()
    cfs_eval = cfs_eval.reindex(cfs.index)

    ## Decoding features
    cfs_decoded = FeatureDecoder(cfs, dataset['discrete_features'], dataset['feature_encoder'])

    ## Predicting counterfactuals
    if cf_label is None:
        cfs_prob = None
        cfs_y = blackbox.predict(cfs)
        return cfs, cfs_decoded, cfs_y, cfs_prob, cfs_eval
    else:
        cfs_y = blackbox.predict(cfs)
        cfs_prob = blackbox.predict_proba(cfs)
        return cfs, cfs_decoded, cfs_y, cfs_prob, cfs_eval

def MOCF(x, blackbox, dataset, X_train, Y_train, probability_thresh=None, cf_label=None, x_range=None, cf_range=None):

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

    if cf_label is None:
        abs_error = np.abs(Y_train-pred_train)
        mae = np.mean(abs_error)
        gt = X_train[np.where(abs_error<=mae)]
        pred_gt = blackbox.predict(gt)
        gt = gt[np.where(np.logical_and(pred_gt>=cf_range[0], pred_gt<=cf_range[1]))]
    else:
        gt = X_train[np.where(pred_train == Y_train)]
        pred_gt = blackbox.predict(gt)
        gt = gt[np.where(pred_gt == cf_label)]

    theta_gt = (gt - mapping_offset) / mapping_scale
    nbrs_gt = NearestNeighbors(n_neighbors=min(len(gt),200), algorithm='kd_tree').fit(theta_gt)

    ## Initialization
    distances, indices = nbrs_gt.kneighbors(theta_x.reshape(1,-1))
    theta_N = theta_gt[indices[0]].copy()
    probability_vec = [0.1,0.2,0.7]

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
    ## Discrete options = {'fix','any',{a set of possible changes]}
    ## Continuous options = {'fix','any','increase','decrease',[a range of possible changes]}


    ############## Breast cancer data set ##################
    # discrete_features = {'age': [0.0, 5.0],
    #                      'menopause': [0.0, 2.0],
    #                      'tumor-size': [0.0, 10.0],
    #                      'inv-node': [0.0, 6.0],
    #                      'node-caps': [0.0, 1.0],
    #                      'deg-malig': [0.0, 2.0],
    #                      'breast': [0.0, 1.0],
    #                      'breast-quad': [0.0, 4.0],
    #                      'irradiat': [0.0, 1.0]}
    # continuous_features = None

    if dataset['name'] == 'breast-cancer':
        desired_actions = {
            'age':('any'),
            'menopause':('any'),
            'tumor-size':('any'),
            'inv-node':('any'),
            'node-caps':('any'),
            'deg-malig':('any'),
            'breast':('any'),
            'breast-quad':('any'),
            'irradiat':('any')
        }
        actions = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions[index] = desired_actions[a]


        ################# Credit card default data set #################
        # discrete_features = {'SEX': [0.0, 1.0],
        #                      'EDUCATION': [0.0, 6.0],
        #                      'MARRIAGE': [0.0, 3.0],
        #                      'PAY_0': [0.0, 10.0],
        #                      'PAY_2': [0.0, 10.0],
        #                      'PAY_3': [0.0, 10.0],
        #                      'PAY_4': [0.0, 10.0],
        #                      'PAY_5': [0.0, 9.0],
        #                      'PAY_6': [0.0, 9.0]}
        #
        # continuous_features = {'LIMIT_BAL': [10000.0, 1000000.0],
        #                         'AGE': [21.0, 79.0],
        #                         'BILL_AMT1': [-165580.0, 964511.0],
        #                         'BILL_AMT2': [-69777.0, 983931.0],
        #                         'BILL_AMT3': [-157264.0, 1664089.0],
        #                         'BILL_AMT4': [-170000.0, 891586.0],
        #                         'BILL_AMT5': [-81334.0, 927171.0],
        #                         'BILL_AMT6': [-339603.0, 961664.0],
        #                         'PAY_AMT1': [0.0, 873552.0],
        #                         'PAY_AMT2': [0.0, 1684259.0],
        #                         'PAY_AMT3': [0.0, 896040.0],
        #                         'PAY_AMT4': [0.0, 621000.0],
        #                         'PAY_AMT5': [0.0, 426529.0],
        #                         'PAY_AMT6': [0.0, 528666.0]}

    elif dataset['name'] == 'credit-card-default':
        desired_actions = {'LIMIT_BAL':('any'),
                            'SEX':('fix'),
                            'EDUCATION':('any'),
                            'MARRIAGE':('fix'),
                            'AGE':('increase'),
                            'PAY_0':({0,2,3,4}),
                            'PAY_2':('any'),
                            'PAY_3':('any'),
                            'PAY_4':('any'),
                            'PAY_5':('any'),
                            'PAY_6':('any'),
                            'BILL_AMT1':('decrease'),
                            'BILL_AMT2':('any'),
                            'BILL_AMT3':('any'),
                            'BILL_AMT4':([-10000,50000]),
                            'BILL_AMT5':('any'),
                            'BILL_AMT6':('any'),
                            'PAY_AMT1':([0,400000]),
                            'PAY_AMT2':('any'),
                            'PAY_AMT3':('any'),
                            'PAY_AMT4':('any'),
                            'PAY_AMT5':('any'),
                            'PAY_AMT6':('increase')
                           }
        actions = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions[index] = desired_actions[a]


    ###################### Adult data set ######################
    # discrete_features = {'work-class': [0.0, 6.0],
    #                     'education': [0.0, 15.0],
    #                     'education-num': [0.0, 15.0],
    #                     'marital-status': [0.0, 6.0],
    #                     'occupation': [0.0, 13.0],
    #                     'relationship': [0.0, 5.0],
    #                     'race': [0.0, 4.0],
    #                     'sex': [0.0, 1.0],
    #                     'native-country': [0.0, 40.0]
    #                    }
    #
    # continuous_features = {'age': [17.0, 90.0],
    #                         'fnlwgt': [13769.0, 1484705.0],
    #                         'capital-gain': [0.0, 99999.0],
    #                         'capital-loss': [0.0, 4356.0],
    #                         'hours-per-week': [1.0, 99.0],
    #                        }

    elif dataset['name'] == 'adult':
        desired_actions = {'age': ('fix'),
                            'work-class': ('any'),
                            'fnlwgt': ('any'),
                            'education': ('fix'),
                            'education-num': ('fix'),
                            'marital-status': ('fix'),
                            'occupation': ('any'),
                            'relationship': ('fix'),
                            'race': ('fix'),
                            'sex': ('fix'),
                            'capital-gain': ('decrease'),
                            'capital-loss': ('increase'),
                            'hours-per-week': ([40,50]),
                            'native-country': ({39,40})
                           }
        actions = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions[index] = desired_actions[a]


    ################## Boston house price data set ####################
    # discrete_features = {'CHAS': [0.0, 1.0]}
    # continuous_features = {'CRIM': [0.00632, 88.9762],
    #                         'ZN': [0.0, 100.0],
    #                         'INDUS': [0.46, 27.74],
    #                         'NOX': [0.385, 0.871],
    #                         'RM': [3.5610000000000004, 8.78],
    #                         'AGE': [2.9, 100.0],
    #                         'DIS': [1.1296, 12.1265],
    #                         'RAD': [1.0, 24.0],
    #                         'TAX': [187.0, 711.0],
    #                         'PTRATIO': [12.6, 22.0],
    #                         'BLACK': [0.32, 396.9],
    #                         'LSTAT': [1.73, 37.97]}

    elif dataset['name'] == 'boston-house-prices':
        desired_actions = {'CRIM': ('any'),
                            'ZN': ('any'),
                            'INDUS': ('any'),
                            'CHAS': ('any'),
                            'NOX': ('any'),
                            'RM': ('any'),
                            'AGE': ('any'),
                            'DIS': ('any'),
                            'RAD': ('any'),
                            'TAX': ('any'),
                            'PTRATIO': ('any'),
                            'BLACK': ('any'),
                            'LSTAT': ('any')
                           }
        actions = [0] * len(x)
        for a in desired_actions:
            index = dataset['feature_names'].index(a)
            actions[index] = desired_actions[a]


    ############### Correlation among features ###############
    # Calculate the correlation/strength-of-association of features in data-set
    # with both categorical and continuous features using:
    # Pearson's R for continuous-continuous cases
    # Correlation Ratio for categorical-continuous cases
    # Cramer's V for categorical-categorical cases
    corr = nominal.associations(X_train, nominal_columns=discrete_indices,theil_u=False, plot=False)['corr']

    ## n-Closest ground truth counterfactual in the training data
    n = 5
    dist = np.asarray([FeatureDistance(x, cf_, feature_range, discrete_indices, continuous_indices) for cf_ in gt])
    closest_ind = np.argsort(dist)[:n]
    for i in range(n):
        theta_cf = (gt[closest_ind[i]] - mapping_offset) / mapping_scale
        print('instnace:', list(gt[closest_ind[i]]), 'probability:',
              blackbox.predict(gt[closest_ind[i]].reshape(1,-1)),
              'cost:',CostFunction(x, theta_x, discrete_indices, continuous_indices,
              mapping_scale, mapping_offset, feature_range, blackbox,
              probability_thresh, cf_label, cf_range, lof_model,
              hdbscan_model, actions, corr, theta_cf))


    ## Parameter setting
    NDIM = len(x)
    NOBJ = 7
    NGEN = 50
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
    f7 =  1.0   # Correlation
    OBJ_W = (f1, f2, f3, f4, f5, f6, f7)

    ## Creating toolbox
    toolbox = SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x, theta_x, discrete_indices, continuous_indices,
                           mapping_scale, mapping_offset, feature_range, blackbox, probability_thresh, cf_label,
                           cf_range, theta_N, probability_vec, lof_model, hdbscan_model, actions, corr)

    ## Running EA
    fronts, pop, record, logbook= RunEA(toolbox, MU, NGEN, CXPB, MUTPB)




    ################## Decision making techniques ######################

    ## Decision making using user-defined priority
    priority = {
        'Distance': 1,
        'Sparsity': 1,
        'Actionable': 1,
        'Connectedness': 0,
        }
    cfs, cfs_decoded, cfs_y, cfs_prob, cfs_eval = ConstructCounterfactuals(x, toolbox, fronts, dataset, mapping_scale,
                                                                           mapping_offset, blackbox, cf_label, priority)
    x_df = pd.DataFrame(data=x.reshape(1,-1), columns=dataset['feature_names']).astype(int)
    x_decoded = FeatureDecoder(x_df, dataset['discrete_features'], dataset['feature_encoder'])

    print('done')
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
