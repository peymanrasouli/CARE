def actionableRecourse(x_org, cf_org, user_preferences):

    action_operation = user_preferences['action_operation']
    action_importance = user_preferences['action_importance']

    cost = []
    idx =  [i for i, op in enumerate(action_operation) if op is not None]
    for i in idx:
        if action_operation[i] == 'fix':
            cost.append(int(cf_org[i] != x_org[i]) * action_importance[i])
        elif action_operation[i] == 'l':
            cost.append(int(cf_org[i] >= x_org[i]) * action_importance[i])
        elif action_operation[i] == 'g':
            cost.append(int(cf_org[i] <= x_org[i]) * action_importance[i])
        elif action_operation[i] == 'ge':
            cost.append(int(cf_org[i] < x_org[i]) * action_importance[i])
        elif action_operation[i] == 'le':
            cost.append(int(cf_org[i] > x_org[i]) * action_importance[i])
        elif type(action_operation[i]) == set:
            cost.append(int(not(cf_org[i] in action_operation[i])) * action_importance[i])
        elif type(action_operation[i]) == list:
            cost.append(int(not(action_operation[i][0] <= cf_org[i] <= action_operation[i][1])) * action_importance[i])
    return sum(cost)