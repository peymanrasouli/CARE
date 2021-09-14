import array
import pandas as pd
from deap import algorithms, base, creator, tools
from utils import *
from sklearn.preprocessing import MinMaxScaler

class CERTIFAI():
    def __init__(self,
                 dataset,
                 predict_fn=None,
                 predict_proba_fn=None,
                 ACTIONABILITY=False,
                 n_cf=5,
                 n_population=50,
                 n_generation=50,
                 hof_size=100,
                 crossover_perc=0.5,
                 mutation_perc=0.2
                 ):

        self.dataset = dataset
        self.feature_names =  dataset['feature_names']
        self.n_features = len(dataset['feature_names'])
        self.feature_width = dataset['feature_width']
        self.continuous_indices = dataset['continuous_indices']
        self.discrete_indices = dataset['discrete_indices']
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.ACTIONABILITY = ACTIONABILITY
        self.n_cf = n_cf
        # self.n_population = n_population
        self.n_population = self.n_features**2
        self.n_generation = n_generation
        self.hof_size = hof_size
        self.crossover_perc = crossover_perc
        self.mutation_perc = mutation_perc


    # defining objective function
    def objectiveFunction(self, cf_theta):

        # constructing counterfactual from the EA decision variables
        cf_theta = np.asarray(cf_theta)
        cf_org = theta2org(cf_theta,  self.featureScaler,  self.dataset)
        cf_ord = org2ord(cf_org,  self.dataset)

        if  self.continuous_indices is not None:
            diff_continuous = np.linalg.norm(cf_ord[self.continuous_indices] - self.x_ord[self.continuous_indices])
        if  self.discrete_indices is not None:
            diff_discrete = sum(cf_ord[self.discrete_indices] != self.x_ord[self.discrete_indices])

        feature_distance = (len(self.continuous_indices)/self.n_features * diff_continuous) + \
                           (len(self.discrete_indices)/self.n_features * diff_discrete)

        return [feature_distance]

    def outcomeConstraint(self, cf_theta):

        # constructing counterfactual from the EA decision variables
        cf_theta = np.asarray(cf_theta)
        cf_org = theta2org(cf_theta,  self.featureScaler,  self.dataset)
        cf_ord = org2ord(cf_org,  self.dataset)
        cf_ohe = ord2ohe(cf_ord, self.dataset)

        cf_pred =  self.predict_fn(cf_ohe.reshape(1, -1))

        return cf_pred

    def actionabilityConstraint(self, cf_theta):

        # constructing counterfactual from the EA decision variables
        cf_theta = np.asarray(cf_theta)
        cf_org = theta2org(cf_theta, self.featureScaler, self.dataset)

        constraint = self.user_preferences['constraint']
        cost = []
        idx = [i for i, c in enumerate(constraint) if c is not None]
        for i in idx:
            if constraint[i] == 'fix':
                cost.append(int(cf_org[i] !=  self.x_org[i]))
            elif constraint[i] == 'l':
                cost.append(int(cf_org[i] >=  self.x_org[i]))
            elif constraint[i] == 'g':
                cost.append(int(cf_org[i] <=  self.x_org[i]))
            elif constraint[i] == 'ge':
                cost.append(int(cf_org[i] <  self.x_org[i]))
            elif constraint[i] == 'le':
                cost.append(int(cf_org[i] >  self.x_org[i]))
            elif type(constraint[i]) == set:
                cost.append(int(not (cf_org[i] in constraint[i])))
            elif type(constraint[i]) == list:
                cost.append(int(not (constraint[i][0] <= cf_org[i] <= constraint[i][1])))
        actionability_distance = sum(cost)

        return actionability_distance

    def checkFeasibility(self, cf_theta):

        # checking the feasibility of counterfactuals w.r.t. outcome and actionability constraints
        if self.ACTIONABILITY:
            cf_pred = self.outcomeConstraint(cf_theta)
            actionability_distance = self.actionabilityConstraint(cf_theta)
            outcome_feasibility = True if cf_pred == self.cf_class else False
            actionability_feasibility = True if actionability_distance == 0 else False
            feasibility = np.logical_and(outcome_feasibility, actionability_feasibility)
        else:
            cf_pred = self.outcomeConstraint(cf_theta)
            feasibility = True if cf_pred == self.cf_class else False

        return feasibility

    def feasibilityDistance(self, cf_theta):

        # measuring the distance of counterfactuals to the feasible solution region
        if self.ACTIONABILITY:
            cf_pred = self.outcomeConstraint(cf_theta)
            actionability_distance = self.actionabilityConstraint(cf_theta)
            if cf_pred != self.cf_class:
                distance = 10.0
            else:
                distance = actionability_distance
        else:
            cf_pred = self.outcomeConstraint(cf_theta)
            distance = 10.0 if cf_pred != self.cf_class else 0.0

        return distance

    def featureScaler(self):

        # creating a scaler for mapping features in equal range
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(self.X_train)

        return feature_scaler


    def fit(self, X_train, Y_train):
        # scaling the feature space from which individuals can be generated
        self.X_train = X_train
        self.Y_train = Y_train
        self.MAD = np.median(np.absolute(X_train - np.median(X_train, axis=0)), axis=0)
        self.featureScaler = self.featureScaler()

    # creating toolbox for optimization algorithm
    def setupToolbox(self):

        # initialization function
        def initialization(n_features):
                return np.random.uniform(0, 1, n_features)

        # creating toolbox
        toolbox = base.Toolbox()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual",  array.array, typecode='d', fitness=creator.FitnessMin)
        toolbox.register("evaluate", self.objectiveFunction)
        toolbox.decorate("evaluate", tools.DeltaPenalty(self.checkFeasibility, 10.0, self.feasibilityDistance))
        toolbox.register("attr_float", initialization, self.n_features)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / self.n_features)
        toolbox.register("select", tools.selTournament,  tournsize=3)

        return toolbox

    # executing the optimization algorithm
    def runEA(self):

        print('CERTIFAI algorithm is running ...')

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        hof = tools.HallOfFame(self.hof_size)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "avg", "max"

        pop = self.toolbox.population(n=self.n_population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(pop=pop, gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self.n_generation):
            offspring = algorithms.varAnd(pop, self.toolbox, self.crossover_perc, self.mutation_perc)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, self.n_population)

            # Compile statistics about the new population
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(pop=pop, gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, hof, record, logbook

    # explain instance using multi-objective counterfactuals
    def explain(self,
                x,
                cf_class='opposite',
                user_preferences=None
                ):

        self.x_ord = x
        self.x_ohe = ord2ohe(x, self.dataset)
        self.x_org = ord2org(x, self.dataset)

        # finding the label of counterfactual instance
        if cf_class is 'opposite':
            x_class = self.predict_fn(self.x_ohe.reshape(1,-1))
            self.cf_class = 1 - x_class[0]
        elif cf_class is 'neighbor':
            x_proba = self.predict_proba_fn(self.x_ohe.reshape(1,-1))
            self.cf_class = np.argsort(x_proba)[0][-2]
        elif cf_class is 'strange':
            x_proba = self.predict_proba_fn(self.x_ohe.reshape(1,-1))
            self.cf_class = np.argsort(x_proba)[0][0]

        self.user_preferences = user_preferences

        # creating toolbox for the counterfactual instance
        self.toolbox = self.setupToolbox()

        # running optimization algorithm for finding counterfactual instances
        pop, hof, record, logbook = self.runEA()

        ## constructing counterfactuals
        cfs_theta = np.asarray([i for i in hof.items])
        cfs_org = theta2org(cfs_theta, self.featureScaler, self.dataset)
        cfs_ord = org2ord(cfs_org, self.dataset)
        cfs_ord = pd.DataFrame(data=cfs_ord, columns=self.feature_names)
        cfs_ord.drop_duplicates(keep="first", inplace=True)

        # selecting the top n_cf counterfactuals
        cfs_ord = cfs_ord.iloc[:self.n_cf,:]

        ## returning the results
        explanations = {'cfs_ord': cfs_ord}

        return explanations