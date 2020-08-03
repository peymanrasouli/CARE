import numpy as np
from nsga2.cost_function import CostFunction
from nsga2.cross_over import Crossover
from nsga2.mutate import Mutate
from nsga2.non_dominated_sorting import NonDominatedSorting
from nsga2.calc_crowding_distance import CalcCrowdingDistance
from nsga2.sort_population import SortPopulation
from matplotlib import pyplot as plt

def NSGA2(x, l_cf, blackbox, dataset, X_train, Y_train, probability_range, MaxIt=100, nPop=100):

    ## Problem definition
    var_size = len(x)   # Decision Variables Matrix Size
    var_min = np.min(dataset['X'], axis=0)  # Lower bound of variables
    var_max = np.max(dataset['X'], axis=0)  # Upper bound of variables

    disc_ind = dataset['discrete_indices']
    cont_ind = dataset['continuous_indices']

    ## GA parameters
    pc = 0.6    # Crossover Percentage
    pm = 0.3    # Mutation Percentage
    co = 0.3    # Crossover Rate
    mu = 0.1   # Mutation Rate
    nc = 2 * round(pc * nPop / 2) # Number of Offsprings
    nm = round(pm * nPop) # Number of Mutants

    ## Initialization
    empty_individual = {
                        'position':[],
                        'cost':[],
                        'sol':[],
                        'rank':[],
                        'dominationSet':[],
                        'dominatedCount':[],
                        'crowdingDistance':[]
                        }

    pop = [empty_individual.copy() for _ in range(nPop)]

    for i in range(nPop):
        # Initialize positions
        position = np.zeros(var_size)
        if disc_ind is not None:
            position[disc_ind] = np.random.randint(var_min[disc_ind], var_max[disc_ind], len(disc_ind))
        if cont_ind is not None:
            position[cont_ind] = np.random.uniform(var_min[cont_ind], var_max[cont_ind], len(cont_ind))
        pop[i]['position'] = position

        # Evaluation
        pop[i]['cost'], pop[i]['sol'] = CostFunction(x,pop[i]['position'], l_cf, var_min, var_max,
                                                     disc_ind, cont_ind , blackbox, probability_range)

    # Non-Dominated Sorting
    pop , F = NonDominatedSorting(pop)

    # Calculate Crowding Distance
    pop = CalcCrowdingDistance(pop,F)

    # Sort Population
    pop, F = SortPopulation(pop)

    ## Main loop
    for it in range(MaxIt):

        # Crossover
        popc_1 =  [empty_individual.copy() for _ in range(int(nc/2))]
        popc_2 =  [empty_individual.copy() for _ in range(int(nc/2))]

        for k in range(int(nc/2)):
            # Select Parents Indices
            i1 = np.random.randint(0,nPop,1)[0]
            i2 = np.random.randint(0,nPop,1)[0]

            # Select Parents
            p1 = pop[i1]
            p2 = pop[i2]

            # Apply crossover
            popc_1[k]['position'], popc_2[k]['position'] = Crossover(p1['position'], p2['position'], co)

            # Evaluate offsprings
            popc_1[k]['cost'], popc_1[k]['sol'] = CostFunction(x, popc_1[k]['position'], l_cf, var_min, var_max,
                                                               disc_ind, cont_ind, blackbox, probability_range)
            popc_2[k]['cost'], popc_2[k]['sol'] = CostFunction(x, popc_2[k]['position'], l_cf, var_min, var_max,
                                                               disc_ind, cont_ind, blackbox, probability_range)

        popc = popc_1 + popc_2

        # Mutation
        popm = [empty_individual.copy() for _ in range(nm)]

        for k in range(nm):
            # Select Parents
            i = np.random.randint(0,nPop,1)[0]
            p = pop[i]

            # Apply mutation
            popm[k]['position']= Mutate(p['position'], mu, var_min, var_max, disc_ind, cont_ind)

            # Evaluate mutant
            popm[k]['cost'], popm[k]['sol'] = CostFunction(x, popm[k]['position'], l_cf, var_min, var_max,
                                                           disc_ind, cont_ind, blackbox, probability_range)

        # Create merged population
        pop = pop+popc+popm

        # Non-Dominated Sorting
        pop, F = NonDominatedSorting(pop)

        # Calculate Crowding Distance
        pop = CalcCrowdingDistance(pop, F)

        # Sort Population
        pop, F = SortPopulation(pop)

        # Truncate
        pop = [pop[i] for i in range(nPop)]

        # Non-Dominated Sorting
        pop, F = NonDominatedSorting(pop)

        # Calculate Crowding Distance
        pop = CalcCrowdingDistance(pop, F)

        # Sort Population
        pop, F = SortPopulation(pop)

        # Store F1
        if F == []:
            return F1
        else:
            F1 = [pop[i] for i in F[0]]

        print('Iteration=',it,'--','Number of F1 members=',len(F1))

        costs = np.asarray([f['cost'] for f in F1])

        plt.plot(costs[:,0],costs[:,1],'r*')
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.draw()
        plt.pause(0.01)
        plt.clf()



