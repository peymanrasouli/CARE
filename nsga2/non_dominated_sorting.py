from nsga2.dominates import Dominates

def NonDominatedSorting(pop):
    nPop = len(pop)

    for i in range(nPop):
        pop[i]['dominationSet'] = []
        pop[i]['dominatedCount'] = 0

    F = [[]]

    for i in range(nPop):
        for j in range(i+1,nPop):
            p = pop[i].copy()
            q = pop[j].copy()

            if Dominates(p,q):
                p['dominationSet'].append(j)
                q['dominatedCount'] =  q['dominatedCount'] +  1

            if Dominates(q,p):
                q['dominationSet'].append(i)
                p['dominatedCount'] = p['dominatedCount'] + 1

            pop[i] = p
            pop[j] = q

        if pop[i]['dominatedCount'] == 0:
            F[0].append(i)
            pop[i]['rank'] = 0

    k = 0

    while True:
        Q = []

        for i in F[k]:
            p = pop[i].copy()

            for j in p['dominationSet']:
                q = pop[j].copy()

                q['dominatedCount'] = q['dominatedCount'] - 1

                if q['dominatedCount'] == 0:
                    Q.append(j)
                    q['rank']= k+1

                pop[j] = q
        if Q == []:
            break

        F.append(Q)

        k += 1

    return pop, F
