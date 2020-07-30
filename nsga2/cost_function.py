
def CostFunction(x_hat):

    ## Objective 1: opposite outcome
    z1 = 0


    ## Objective 2: proximity
    z2 = 0


    ## Objective 3: connectedness
    z3 = 0


    ## Objective 4: actionable
    z4 = 0


    ## Objective 5: sparsity
    z5 = 0




    z = [z1,z2,z3,z4,z5]

    sol= {
        'z1': z1,
        'z2': z2,
        'z3': z3,
        'z4': z4,
        'z5': z5,
    }

    return z , sol