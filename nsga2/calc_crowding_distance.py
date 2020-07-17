import numpy as np
import math

def CalcCrowdingDistance(pop,F):
    nF = len(F)

    for k in range(nF):
        costs = np.asarray([pop[i]['cost'] for i in F[k]])

        nObj = costs.shape[1]

        n = len(F[k])

        d = np.zeros([n,nObj])

        for j in range(nObj):

            so = np.argsort(costs[:,j])
            cj = costs[so,j]

            d[so[0],j] = math.inf

            for i in range(1,n-1):
                d[so[i],j] = np.abs(cj[i+1]-cj[i-1])/np.abs(cj[0]-cj[-1])

            d[so[-1],j] = math.inf

        for i in range(n):
            pop[F[k][i]]['crowdingDistance'] = np.sum(d[i,:])

    return pop