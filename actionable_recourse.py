def ActionableRecourse(x, cf, actions):
    cost = []
    idx =  [i for i, e in enumerate(actions) if e != 'any']
    for i in idx:
        if actions[i] == 'fix':
            cost.append(int(cf[i] != x[i]))
        if actions[i] == 'increase':
            cost.append(int(cf[i] < x[i]))
        if actions[i] == 'decrease':
            cost.append(int(cf[i] > x[i]))
        elif type(actions[i]) == set:
            cost.append(int(not(cf[i] in actions[i])))
        elif type(actions[i]) == list:
            cost.append(int(not(actions[i][0] <= cf[i] <= actions[i][1])))
    return sum(cost)