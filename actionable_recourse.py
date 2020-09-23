def ActionableRecourse(x_bb, cf_bb, action_operation, action_priority):
    cost = []
    idx =  [i for i, op in enumerate(action_operation) if op is not None]
    for i in idx:
        if action_operation[i] == 'fix':
            cost.append(int(cf_bb[i] != x_bb[i]) * action_priority[i])
        if action_operation[i] == 'increase':
            cost.append(int(cf_bb[i] < x_bb[i]) * action_priority[i])
        if action_operation[i] == 'decrease':
            cost.append(int(cf_bb[i] > x_bb[i]) * action_priority[i])
        elif type(action_operation[i]) == set:
            cost.append(int(not(cf_bb[i] in action_operation[i])) * action_priority[i])
        elif type(action_operation[i]) == list:
            cost.append(int(not(action_operation[i][0] <= cf_bb[i] <= action_operation[i][1])) * action_priority[i])
    return sum(cost)