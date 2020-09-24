import array
import numpy as np
import pandas as pd
from math import *
from deap import algorithms, base, creator, tools
from cost_function import CostFunction
import matplotlib.pyplot as plt
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

def SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x_bb, x_theta, discrete_indices, continuous_indices,
                 feature_encoder, feature_scaler, ea_scaler, feature_width, blackbox, probability_thresh,
                 cf_label, cf_range, nbrs_theta, selection_probability, lof_model, hdbscan_model, action_operation,
                 action_priority, corr_models):

    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=OBJ_W)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
    toolbox.register("evaluate", CostFunction, x_bb, x_theta, discrete_indices, continuous_indices, feature_encoder,
                     feature_scaler, ea_scaler, feature_width, blackbox, probability_thresh, cf_label, cf_range,
                     lof_model, hdbscan_model, action_operation, action_priority, corr_models)
    toolbox.register("attr_float", Initialization, BOUND_LOW, BOUND_UP, NDIM, x_theta, nbrs_theta, selection_probability)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
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

def ConstructCounterfactuals(dataset, toolbox, fronts, ea_scaler, constraints):
    ## Constructing counterfactuals
    pop = []
    evaluation = []
    for f in fronts:
        for ind in f:
            pop.append(np.asarray(ind))
            evaluation.append(np.asarray(toolbox.evaluate(ind)))
    pop = np.asarray(pop)
    evaluation = np.asarray(evaluation)

    discrete_indices = dataset['discrete_indices']
    solutions = pop.copy()
    solutions[:,discrete_indices] = ea_scaler.inverse_transform(solutions[:,discrete_indices])
    solutions[:,discrete_indices] = np.rint(solutions[:,discrete_indices])

    col = dataset['feature_names']
    cfs = pd.DataFrame(data=solutions, columns=col)

    col = ['f'+str(i+1) for i in range(evaluation.shape[1])]
    cfs_eval = pd.DataFrame(data=evaluation, columns=col)

    drop_ind = []
    for c in constraints:
        f = cfs_eval[c[0]]
        op = c[1]
        val = c[2]
        ind = np.where(np.logical_not(op(f,val)))[0]
        drop_ind.append(ind)

    drop_ind = np.concatenate(drop_ind)
    drop_ind = np.unique(drop_ind)

    cfs.drop(drop_ind, inplace=True)
    cfs_eval.drop(drop_ind, inplace=True)

    cfs = cfs.drop_duplicates()
    cfs_eval = cfs_eval.reindex(cfs.index)

    cfs.reset_index(drop=True, inplace=True)
    cfs_eval.reset_index(drop=True, inplace=True)

    return cfs, cfs_eval

def MOCFExplainer(x_bb, blackbox, dataset, X_train, Y_train, probability_thresh=None, cf_label=None, x_range=None, cf_range=None):
    ## Reading dataset information
    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_width = feature_max - feature_min

    # Scaling data to (0,1) for EA
    ea_scaler = MinMaxScaler()
    ea_scaler.fit(X_train[:,discrete_indices])
    X_train_theta = X_train.copy()
    X_train_theta[:,discrete_indices] = ea_scaler.transform(X_train_theta[:,discrete_indices])

    ## KNN model of correctly classified samples same class as counter-factual
    pred_train = blackbox.predict(X_train)
    if cf_label is None:
        abs_error = np.abs(Y_train-pred_train)
        mae = np.mean(abs_error)
        gt = X_train[np.where(abs_error<=mae)]
        pred_gt = blackbox.predict(gt)
        gt = gt[np.where(np.logical_and(pred_gt>=cf_range[0], pred_gt<=cf_range[1]))]
        gt_theta = gt.copy()
        gt_theta[:,discrete_indices] = ea_scaler.transform(gt[:,discrete_indices])
    else:
        gt = X_train[np.where(pred_train == Y_train)]
        pred_gt = blackbox.predict(gt)
        gt = gt[np.where(pred_gt == cf_label)]
        gt_theta = gt.copy()
        gt_theta[:, discrete_indices] = ea_scaler.transform(gt[:, discrete_indices])

    K_nbrs = min(500, len(gt_theta))
    gt_nbrModel = NearestNeighbors(n_neighbors=K_nbrs, algorithm='kd_tree').fit(gt_theta)

    ## Creating local outlier factor model for proximity function
    lof_model = LocalOutlierFactor(n_neighbors=1, novelty=True)
    lof_model.fit(gt_theta)

    ## Creating hdbscan clustering model for connectedness function
    dist = pairwise_distances(gt_theta, metric='minkowski')
    dist[np.where(dist==0)] = np.inf
    epsilon = np.max(np.min(dist,axis=0))
    hdbscan_model = hdbscan.HDBSCAN(min_samples=2, cluster_selection_epsilon=float(epsilon),
                                    metric='minkowski', p=2, prediction_data=True).fit(gt_theta)

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

        action_operation = [None] * len(x_bb)
        action_priority = [None] * len(x_bb)
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

        action_operation = [None] * len(x_bb)
        action_priority = [None] * len(x_bb)
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

        action_operation = [None] * len(x_bb)
        action_priority = [None] * len(x_bb)
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

        action_operation = [None] * len(x_bb)
        action_priority = [None] * len(x_bb)
        for p in preferences:
            index = dataset['feature_names'].index(p)
            action_operation[index] = preferences[p][0]
            action_priority[index] = preferences[p][1]

    ## Feature correlation modeling
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
    x_theta = x_bb.copy()
    x_theta[discrete_indices] = ea_scaler.transform(x_theta[discrete_indices].reshape(1, -1)).ravel()

    distances, indices = gt_nbrModel.kneighbors(x_theta.reshape(1, -1))
    nbrs_theta = gt_theta[indices[0]].copy()

    selection_probability = {'x': 0.1, 'neighbor':0.2, 'random':0.7}


    # Objective functions || -1.0: cost function | 1.0: fitness function
    f1 = -1.0   # Prediction Distance
    f2 = -1.0   # Feature Distance
    f3 =  1.0   # Proximity
    f4 = -1.0   # Actionable Recourse
    f5 = -1.0   # Sparsity
    f6 =  1.0   # Connectedness
    f7 =  -1.0   # Correlation
    OBJ_W = (f1, f2, f3, f4, f5, f6, f7)

    # EA parameters
    NDIM = len(x_bb)
    NOBJ = len(OBJ_W)
    NGEN = 30
    CXPB = 0.5
    MUTPB = 0.2
    P = 6
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    BOUND_LOW, BOUND_UP = 0, 1

    # Creating toolbox for the EA
    feature_encoder = dataset['feature_encoder']
    feature_scaler = dataset['feature_scaler']
    toolbox = SetupToolbox(NDIM, NOBJ, P, BOUND_LOW, BOUND_UP, OBJ_W, x_bb, x_theta, discrete_indices, continuous_indices,
                           feature_encoder, feature_scaler, ea_scaler, feature_width, blackbox, probability_thresh,
                           cf_label, cf_range, nbrs_theta, selection_probability, lof_model, hdbscan_model, action_operation,
                           action_priority, corr_models)

    ## Running EA
    fronts, pop, record, logbook= RunEA(toolbox, MU, NGEN, CXPB, MUTPB)

    ## Applying constraints and constructing counter-factuals
    constraints = [('f1',np.less_equal,0), ('f3',np.equal,1), ('f6',np.greater,0)]
    cfs, cfs_eval = ConstructCounterfactuals(dataset, toolbox, fronts, ea_scaler, constraints)

    ## Recovering original data


    ## Returning the results
    output = {'cfs': cfs,
              'cfs_eval': cfs_eval,
              'fronts': fronts,
              'pop': pop,
              'toolbox': toolbox,
              'ea_scaler': ea_scaler
    }

    return output

