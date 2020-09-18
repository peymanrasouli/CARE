def ActionableRecourse(x, cf, action_operation, action_priority):
    cost = []
    idx =  [i for i, e in enumerate(action_operation) if e != 'any']
    for i in idx:
        if action_operation[i] == 'fix':
            cost.append(int(cf[i] != x[i]) * action_priority[i])
        if action_operation[i] == 'increase':
            cost.append(int(cf[i] < x[i]) * action_priority[i])
        if action_operation[i] == 'decrease':
            cost.append(int(cf[i] > x[i]) * action_priority[i])
        elif type(action_operation[i]) == set:
            cost.append(int(not(cf[i] in action_operation[i])) * action_priority[i])
        elif type(action_operation[i]) == list:
            cost.append(int(not(action_operation[i][0] <= cf[i] <= action_operation[i][1])) * action_priority[i])
    return sum(cost)