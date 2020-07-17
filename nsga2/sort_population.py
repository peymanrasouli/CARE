import numpy as np

def SortPopulation(pop):

    # Sort Based on Crowding Distance
    crowdingDistances = np.asarray([pop[i]['crowdingDistance'] for i in range(len(pop))])
    CDSO = np.argsort(crowdingDistances)[::-1]
    pop = [pop[i] for i in CDSO]

    # Sort Based on Rank
    ranks = np.asarray([pop[i]['rank'] for i in range(len(pop))])
    RSO = np.argsort(ranks)
    pop = [pop[i] for i in RSO]

    # Update Fronts
    maxRank = np.max(ranks)
    F = [[] for _ in range(maxRank)]
    for r in range(maxRank):
        F[r] = list(np.where(ranks == r)[0])

    return pop, F