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
from sklearn.metrics import pairwise_distances, f1_score, r2_score
from dython import nominal
import hdbscan
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

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
                 mapping_scale, mapping_offset, feature_range, feature_width, blackbox, probability_thresh, cf_label,
                 cf_range, theta_N, probability_vec, lof_model, hdbscan_model, action_operation, action_priority,
                 corr_models, corr):
    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=OBJ_W)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register("evaluate", CostFunction, x, theta_x, discrete_indices, continuous_indices, mapping_scale,
                     mapping_offset, feature_range, feature_width, blackbox, probability_thresh, cf_label, cf_range,
                     lof_model, hdbscan_model, action_operation, action_priority, corr_models, corr)
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
                                                          'Sparsity', 'Connectedness', 'Correlation', 'Response'])
    else:
        label = blackbox.predict(cfs)
        prob = blackbox.predict_proba(cfs)[:,cf_label]
        evaluation = np.c_[evaluation, label, prob]
        cfs_eval = pd.DataFrame(data=evaluation, columns=['Prediction', 'Distance', 'Proximity', 'Actionable',
                                                          'Sparsity', 'Connectedness', 'Correlation', 'Label', 'Probability'])

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
    feature_width = feature_max - feature_min
    feature_range = [feature_min,feature_max]

    ## Linear mapping
    theta_min = -1
    theta_max = 1
    mapping_scale = (feature_max - feature_min) / (theta_max - theta_min)
    mapping_offset = -theta_min * (feature_max - feature_min) / (theta_max - theta_min) + feature_min
    theta_x = (x - mapping_offset) / mapping_scale
    theta_train = (X_train - mapping_offset) / mapping_scale

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

    # preferences = {'age': (operation, priority),
    #                'menopause': (operation, priority),
    #                'tumor-size': (operation, priority),
    #                'inv-node': (operation, priority),
    #                'node-caps': (operation, priority),
    #                'deg-malig': (operation, priority),
    #                'breast': (operation, priority),
    #                'breast-quad': (operation, priority),
    #                'irradiat': (operation, priority),
    #                }

    if dataset['name'] == 'breast-cancer':
        preferences = {}

        action_operation = [None] * len(x)
        action_priority = [None] * len(x)
        for p in preferences:
            index = dataset['feature_names'].index(p)
            action_operation[index] = preferences[p][0]
            action_priority[index] = preferences[p][1]


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

        # preferences = {'LIMIT_BAL':(operation, priority),
        #                 'SEX':(operation, priority),
        #                 'EDUCATION':(operation, priority),
        #                 'MARRIAGE':(operation, priority),
        #                 'AGE':(operation, priority),
        #                 'PAY_0':(operation, priority),
        #                 'PAY_2':(operation, priority),
        #                 'PAY_3':(operation, priority),
        #                 'PAY_4':(operation, priority),
        #                 'PAY_5':(operation, priority),
        #                 'PAY_6':(operation, priority),
        #                 'BILL_AMT1':(operation, priority),
        #                 'BILL_AMT2':(operation, priority),
        #                 'BILL_AMT3':(operation, priority),
        #                 'BILL_AMT4':(operation, priority),
        #                 'BILL_AMT5':(operation, priority),
        #                 'BILL_AMT6':(operation, priority),
        #                 'PAY_AMT1':(operation, priority),
        #                 'PAY_AMT2':(operation, priority),
        #                 'PAY_AMT3':(operation, priority),
        #                 'PAY_AMT4':(operation, priority),
        #                 'PAY_AMT5':(operation, priority),
        #                 'PAY_AMT6':(operation, priority),
        #                }

    elif dataset['name'] == 'credit-card-default':
        preferences = {}

        action_operation = [None] * len(x)
        action_priority = [None] * len(x)
        for p in preferences:
            index = dataset['feature_names'].index(p)
            action_operation[index] = preferences[p][0]
            action_priority[index] = preferences[p][1]

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

    # preferences = {'age': (operation, priority),
    #                 'work-class': (operation, priority),
    #                 'fnlwgt': (operation, priority),
    #                 'education': (operation, priority),
    #                 'education-num': (operation, priority),
    #                 'marital-status': (operation, priority),
    #                 'occupation':(operation, priority),
    #                 'relationship': (operation, priority),
    #                 'race': (operation, priority),
    #                 'sex':(operation, priority),
    #                 'capital-gain': (operation, priority),
    #                 'capital-loss': (operation, priority),
    #                 'hours-per-week': (operation, priority),
    #                 'native-country': (operation, priority),
    #                }


    elif dataset['name'] == 'adult':
        # preferences = {'age': ('increase', 5),
        #                 'marital-status': ('fix', 6),
        #                 'occupation':({0,1,2,3,4,5}, 4),
        #                 'relationship': ('fix', 7),
        #                 'race': ('fix', 10),
        #                 'sex':('fix', 9),
        #                 'capital-gain': ([0,10000], 1),
        #                 'capital-loss': ([0,2000], 2),
        #                 'hours-per-week': ('increase', 3),
        #                 'native-country': ('fix', 8),
        #                }

        preferences = {}

        action_operation = [None] * len(x)
        action_priority = [None] * len(x)
        for p in preferences:
            index = dataset['feature_names'].index(p)
            action_operation[index] = preferences[p][0]
            action_priority[index] = preferences[p][1]

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

    # preferences = {'CRIM': (operation, priority),
    #                 'ZN': (operation, priority),
    #                 'INDUS': (operation, priority),
    #                 'CHAS': (operation, priority),
    #                 'NOX': (operation, priority),
    #                 'RM': (operation, priority),
    #                 'AGE': (operation, priority),
    #                 'DIS': (operation, priority),
    #                 'RAD': (operation, priority),
    #                 'TAX': (operation, priority),
    #                 'PTRATIO': (operation, priority),
    #                 'BLACK': (operation, priority),
    #                 'LSTAT': (operation, priority),
    #                }

    elif dataset['name'] == 'boston-house-prices':
        preferences = {}

        action_operation = [None] * len(x)
        action_priority = [None] * len(x)
        for p in preferences:
            index = dataset['feature_names'].index(p)
            action_operation[index] = preferences[p][0]
            action_priority[index] = preferences[p][1]

    ############### Correlation among features ###############
    # Calculate the correlation/strength-of-association of features in data-set
    # with both categorical and continuous features using:
    # Pearson's R for continuous-continuous cases
    # Correlation Ratio for categorical-continuous cases
    # Cramer's V for categorical-categorical cases
    corr = nominal.associations(X_train, nominal_columns=discrete_indices,theil_u=False, plot=False)['corr']
    corr = corr.to_numpy()
    corr[np.diag_indices(corr.shape[0])] = 0
    corr_thresh = 0.2
    corr_features = np.where(abs(corr) > corr_thresh)
    corr_ = np.zeros(corr.shape)
    corr_[corr_features] = 1

    ## Linear models as correlation models
    val_idx = int(0.7 * len(X_train))
    corr_models = []
    for f in range(len(corr_)):
        inputs = np.where(corr_[f,:] == 1)[0]
        if len(inputs) > 0:
            if f in discrete_indices:
                model = DecisionTreeClassifier()
                model.fit(theta_train[0:val_idx, inputs], X_train[0:val_idx, f])
                score = f1_score(X_train[val_idx:, f], model.predict(theta_train[val_idx:, inputs]), average='weighted')
                if score > 0.7:
                    corr_models.append({'feature': f, 'inputs': inputs, 'model': model, 'score': score})
            elif f in continuous_indices:
                model = DecisionTreeRegressor()
                model.fit(theta_train[0:val_idx,inputs], X_train[0:val_idx,f])
                score = r2_score(X_train[val_idx:,f], model.predict(theta_train[val_idx:,inputs]))
                if score > 0.7:
                    corr_models.append({'feature':f, 'inputs':inputs, 'model':model, 'score':score})

    ## n-Closest ground truth counterfactual in the training data
    n = 5
    dist = np.asarray([FeatureDistance(x, cf_, feature_width, discrete_indices, continuous_indices) for cf_ in gt])
    closest_ind = np.argsort(dist)[:n]
    for i in range(n):
        theta_cf = (gt[closest_ind[i]] - mapping_offset) / mapping_scale
        print('instnace:', list(gt[closest_ind[i]]), 'probability:', blackbox.predict(gt[closest_ind[i]].reshape(1,-1)),
              'cost:',CostFunction(x, theta_x, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
              feature_range, feature_width, blackbox, probability_thresh, cf_label, cf_range, lof_model, hdbscan_model,
              action_operation, action_priority, corr_models, corr, theta_cf))


    ## Parameter setting
    ##  Objective functions || -1.0: cost function | 1.0: fitness function
    f1 = -1.0   # Prediction Distance
    f2 = -1.0   # Feature Distance
    f3 =  1.0   # Proximity
    f4 = -1.0   # Actionable Recourse
    f5 = -1.0   # Sparsity
    f6 =  1.0   # Connectedness
    f7 =  -1.0   # Correlation
    OBJ_W = (f1, f2, f3, f4, f5, f6, f7)

    NDIM = len(x)
    NOBJ = len(OBJ_W)
    NGEN = 50
    CXPB = 0.5
    MUTPB = 0.2
    P = 6
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    BOUND_LOW, BOUND_UP = theta_min, theta_max

    ## Creating toolbox
    toolbox = SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x, theta_x, discrete_indices, continuous_indices,
                           mapping_scale, mapping_offset, feature_range, feature_width, blackbox, probability_thresh,
                           cf_label, cf_range, theta_N, probability_vec, lof_model, hdbscan_model, action_operation,
                           action_priority, corr_models, corr)

    ## Running EA
    fronts, pop, record, logbook= RunEA(toolbox, MU, NGEN, CXPB, MUTPB)




    ################## Decision making techniques ######################

    ## Decision making using user-defined priority
    priority = {
        'Sparsity': 1,
        'Distance': 1,
        'Correlation': 1,
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
