from nsga2.nsga2 import NSGA2

def MOCF(x, blackbox, dataset, probability_range):

    l_x = blackbox.predict(x.reshape(1,-1))
    l_cf = int(1 - l_x)

    MaxIt = 100
    nPop = 200

    F1 = NSGA2(x, l_cf, blackbox, dataset, probability_range, MaxIt=MaxIt, nPop=nPop)



