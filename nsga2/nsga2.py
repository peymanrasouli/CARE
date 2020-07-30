import numpy as np
from nsga2.cost_function import CostFunction
from nsga2.cross_over import Crossover
from nsga2.mutate import Mutate
from nsga2.non_dominated_sorting import NonDominatedSorting
from nsga2.calc_crowding_distance import CalcCrowdingDistance
from nsga2.sort_population import SortPopulation
from matplotlib import pyplot as plt

def nsga2(X, blackbox, x, MaxIt=100, nPop=100):

    ## Problem definition
    VarSize = X.shape[1]   # Decision Variables Matrix Size
    VarMin = np.min(X, axis=0)  # Lower bound of variables
    VarMax = np.max(X, axis=0)  # Upper bound of variables

    # % Number of Objective Functions
    nObj = 2

    MAD =  np.median(np.absolute(X - np.median(X, axis=0)), axis=0)
    
    ## GA parameters
    pc = 0.8    # Crossover Percentage
    pm = 0.3    # Mutation Percentage
    gamma = 0.05
    mu = 0.02   # Mutation Rate
    beta = 8    # Selection Pressure
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
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, VarSize)
        # Evaluation
        pop[i]['cost'], pop[i]['sol'] = CostFunction(pop[i]['position'])

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
            popc_1[k]['position'], popc_2[k]['position'] = Crossover(p1['position'], p2['position'],gamma,VarMin,VarMax)

            # Evaluate offsprings
            popc_1[k]['cost'], popc_1[k]['sol'] = CostFunction(popc_1[k]['position'])
            popc_2[k]['cost'], popc_2[k]['sol'] = CostFunction(popc_2[k]['position'])

        popc = popc_1 + popc_2

        # Mutation
        popm = [empty_individual.copy() for _ in range(nm)]

        for k in range(nm):
            # Select Parents
            i = np.random.randint(0,nPop,1)[0]
            p = pop[i]

            # Apply mutation
            popm[k]['position']= Mutate(p['position'], mu, VarMin, VarMax)

            # Evaluate mutant
            popm[k]['cost'], popm[k]['sol'] = CostFunction(popm[k]['position'])

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
            cf_samples = np.asarray([np.round(f1['position']) for f1 in F1])
            return cf_samples
        else:
            F1 = [pop[i] for i in F[0]]

        # print('Iteration=',it,'--','Number of F1 members=',len(F1))

        costs = np.asarray([f['cost'] for f in F1])

        plt.plot(costs[:,0],costs[:,1],'r*')
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.draw()
        plt.pause(0.01)
        plt.clf()




